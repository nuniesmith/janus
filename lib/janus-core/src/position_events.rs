//! Position event ingress (JFLOW-C).
//!
//! Receives live position snapshots from the execution side (Ruby / fks),
//! defines the wire shape, validates inputs, and computes a guidance hint
//! (hold / reduce / exit) the producer can act on. The receiving HTTP
//! handler lives in `janus-api`; persistence and brain-pipeline-driven
//! guidance refinement are follow-ups.

use crate::market::Side;
use crate::optimized_params::OptimizedParams;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// A snapshot of an open position pushed by the execution side.
///
/// Shape is the minimum needed for downstream guidance. Add fields here when
/// a consumer (guidance engine / memory store) actually reads them.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionEvent {
    /// Trading pair / instrument (e.g. "BTC-USD").
    pub symbol: String,
    /// Direction: Buy = long, Sell = short.
    pub side: Side,
    /// Position size in base units (always positive; direction is in `side`).
    pub qty: f64,
    /// Average fill price the position was opened at.
    pub entry_price: f64,
    /// Mark / last price used to compute `pnl_unrealized`.
    pub current_price: f64,
    /// Unrealized P&L in quote currency (signed).
    pub pnl_unrealized: f64,
    /// Optional client position id for correlation across repeated pushes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub position_id: Option<String>,
    /// Optional JanusAI session id (groups positions under one run).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// Optional volatility hint: recent ATR expressed as a percentage of
    /// `current_price` (e.g. `1.5` = ATR is 1.5% of price). When present,
    /// guidance widens the stop-loss to sit outside normal ATR-sized
    /// noise. Producers that don't track ATR simply omit it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub atr_pct: Option<f64>,
}

impl PositionEvent {
    /// Reject obviously malformed events at the boundary.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.symbol.is_empty() {
            return Err("symbol is empty");
        }
        if !self.qty.is_finite() || self.qty <= 0.0 {
            return Err("qty must be positive and finite");
        }
        if !self.entry_price.is_finite() || self.entry_price <= 0.0 {
            return Err("entry_price must be positive and finite");
        }
        if !self.current_price.is_finite() || self.current_price <= 0.0 {
            return Err("current_price must be positive and finite");
        }
        if !self.pnl_unrealized.is_finite() {
            return Err("pnl_unrealized must be finite");
        }
        if let Some(atr) = self.atr_pct
            && (!atr.is_finite() || atr < 0.0)
        {
            return Err("atr_pct must be a non-negative finite percentage");
        }
        Ok(())
    }

    /// Unrealized P&L as a ratio of entry notional (`entry_price * qty`).
    ///
    /// Returns `None` when notional is non-positive. Sign-agnostic — the
    /// sign of `pnl_unrealized` already encodes direction for Buy/Sell, so
    /// `+0.05` is a 5% gain whether long or short.
    pub fn pnl_ratio(&self) -> Option<f64> {
        let notional = self.entry_price * self.qty;
        (notional > 0.0).then(|| self.pnl_unrealized / notional)
    }
}

/// A terminal event: the producer closed a position. Carries the realized
/// outcome so Janus can record what actually happened against the guidance it
/// advised while the position was open. Sent to `POST /api/v1/positions/close`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionClose {
    /// Trading pair / instrument (e.g. "BTC-USD").
    pub symbol: String,
    /// Direction the (now-closed) position was held in.
    pub side: Side,
    /// Position size in base units (always positive).
    pub qty: f64,
    /// Average fill price the position was opened at.
    pub entry_price: f64,
    /// Average fill price the position was closed at.
    pub exit_price: f64,
    /// Realized P&L in quote currency (signed).
    pub pnl_realized: f64,
    /// Optional realized risk-reward ratio for this trade, when the producer
    /// tracks it. Fed straight into affinity learning.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rr_ratio: Option<f64>,
    /// Optional strategy name that opened this position. Required for the
    /// outcome to be fed back into the per-`(strategy, asset)` affinity
    /// tracker; omitted closes are still persisted, just not recorded live.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strategy: Option<String>,
    /// Client position id, used to correlate with the open snapshots so the
    /// outcome can be joined with this position's guidance history.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub position_id: Option<String>,
    /// Optional JanusAI session id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
}

impl PositionClose {
    /// Reject malformed close events at the boundary.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.symbol.is_empty() {
            return Err("symbol is empty");
        }
        if !self.qty.is_finite() || self.qty <= 0.0 {
            return Err("qty must be positive and finite");
        }
        if !self.entry_price.is_finite() || self.entry_price <= 0.0 {
            return Err("entry_price must be positive and finite");
        }
        if !self.exit_price.is_finite() || self.exit_price <= 0.0 {
            return Err("exit_price must be positive and finite");
        }
        if !self.pnl_realized.is_finite() {
            return Err("pnl_realized must be finite");
        }
        Ok(())
    }

    /// Realized P&L as a ratio of entry notional. `None` when notional is
    /// non-positive (impossible after [`validate`](Self::validate)).
    pub fn realized_ratio(&self) -> Option<f64> {
        let notional = self.entry_price * self.qty;
        (notional > 0.0).then(|| self.pnl_realized / notional)
    }
}

/// Coarse win/loss classification of a closed trade.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OutcomeResult {
    Win,
    Loss,
    Breakeven,
}

impl OutcomeResult {
    /// Classify from a realized P&L ratio (treats `|ratio| < 1e-9` as flat).
    pub fn from_ratio(ratio: f64) -> Self {
        if ratio > 1e-9 {
            Self::Win
        } else if ratio < -1e-9 {
            Self::Loss
        } else {
            Self::Breakeven
        }
    }

    /// Lowercase wire name (matches the serde representation).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Win => "win",
            Self::Loss => "loss",
            Self::Breakeven => "breakeven",
        }
    }
}

/// A closed-trade record joining a [`PositionClose`] with the guidance
/// history accumulated in [`PositionState`] while the position was open.
///
/// This is the "outcome" Janus captures so the guidance engine can be
/// evaluated and tuned. Downstream, the JanusAI service compacts these into
/// `janus_memories` (fks repo). Fields sourced from the tracker are `None`
/// when the position was never tracked (no `position_id`, or no snapshots
/// arrived before the close).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionOutcome {
    pub symbol: String,
    pub side: Side,
    pub qty: f64,
    pub entry_price: f64,
    pub exit_price: f64,
    pub pnl_realized: f64,
    /// Realized P&L as a ratio of entry notional.
    pub realized_ratio: f64,
    /// Win / loss / breakeven from the sign of `realized_ratio`.
    pub result: OutcomeResult,
    /// Realized risk-reward ratio, when the producer reported one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rr_ratio: Option<f64>,
    /// Strategy that opened the position, when reported. Affinity recording
    /// is skipped when this is absent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strategy: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub position_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// Highest unrealized-P&L ratio seen while the position was open.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peak_pnl_ratio: Option<f64>,
    /// Number of open snapshots observed (0 if the position was untracked).
    pub samples: u64,
    /// The last guidance action Janus advised for this position.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_guidance: Option<GuidanceAction>,
    /// Seconds the position was reported open.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_in_position_secs: Option<f64>,
}

impl PositionOutcome {
    /// Build an outcome from a close event joined with the position's
    /// accumulated [`PositionState`] (when it was tracked).
    pub fn from_close(close: &PositionClose, state: Option<&PositionState>) -> Self {
        let realized_ratio = close.realized_ratio().unwrap_or(0.0);
        Self {
            symbol: close.symbol.clone(),
            side: close.side,
            qty: close.qty,
            entry_price: close.entry_price,
            exit_price: close.exit_price,
            pnl_realized: close.pnl_realized,
            realized_ratio,
            result: OutcomeResult::from_ratio(realized_ratio),
            rr_ratio: close.rr_ratio,
            strategy: close.strategy.clone(),
            position_id: close.position_id.clone(),
            session_id: close.session_id.clone(),
            peak_pnl_ratio: state.map(|s| s.peak_pnl_ratio),
            samples: state.map_or(0, |s| s.samples),
            last_guidance: state.map(|s| s.last_action),
            time_in_position_secs: state.map(|s| s.time_in_position().as_secs_f64()),
        }
    }

    /// Whether this trade closed in profit (a `Win`). `Breakeven` and `Loss`
    /// both count as not-a-winner for affinity purposes.
    pub fn is_winner(&self) -> bool {
        self.result == OutcomeResult::Win
    }
}

/// Guidance action returned to the execution side for an open position.
///
/// Producers (Ruby / fks) are free to ignore this — it is advisory. The
/// recommended interpretation:
///
/// - [`Hold`](GuidanceAction::Hold): no change suggested.
/// - [`Reduce`](GuidanceAction::Reduce): trim size (typically take partial
///   profit), but don't close.
/// - [`Exit`](GuidanceAction::Exit): close the position immediately.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GuidanceAction {
    Hold,
    Reduce,
    Exit,
}

impl GuidanceAction {
    /// Lowercase wire name (matches the serde representation).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Hold => "hold",
            Self::Reduce => "reduce",
            Self::Exit => "exit",
        }
    }
}

/// Advisory guidance returned alongside the receive acknowledgement.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Guidance {
    pub action: GuidanceAction,
    pub reason: String,
}

impl Guidance {
    fn hold(reason: impl Into<String>) -> Self {
        Self {
            action: GuidanceAction::Hold,
            reason: reason.into(),
        }
    }
    fn reduce(reason: impl Into<String>) -> Self {
        Self {
            action: GuidanceAction::Reduce,
            reason: reason.into(),
        }
    }
    fn exit(reason: impl Into<String>) -> Self {
        Self {
            action: GuidanceAction::Exit,
            reason: reason.into(),
        }
    }
}

/// Threshold ratios used by [`compute_guidance`]. Decoupled from
/// [`OptimizedParams`] so callers can supply optimizer-derived values when
/// available and fall back to conservative defaults otherwise.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GuidanceThresholds {
    /// Negative ratio of notional at which to exit on a loss (e.g. `-0.02`
    /// = exit when unrealized loss reaches 2% of `entry_price * qty`).
    pub stop_loss_ratio: f64,
    /// Positive ratio of notional at which to reduce on profit (e.g. `0.05`
    /// = trim when unrealized gain reaches 5% of `entry_price * qty`).
    pub take_profit_ratio: f64,
}

impl Default for GuidanceThresholds {
    fn default() -> Self {
        Self {
            stop_loss_ratio: -0.02,
            take_profit_ratio: 0.05,
        }
    }
}

impl GuidanceThresholds {
    /// Derive thresholds from optimizer-tuned params.
    ///
    /// Both `stop_loss_ratio` and `take_profit_ratio` are now learnable
    /// via [`OptimizedParams::stop_loss_pct`] and
    /// [`OptimizedParams::take_profit_pct`] respectively.
    /// `stop_loss_pct` is stored as a positive percentage (e.g. `2.0`)
    /// and converted to a negative ratio here (`-0.02`).
    pub fn from_optimized_params(params: &OptimizedParams) -> Self {
        Self {
            stop_loss_ratio: -(params.stop_loss_pct / 100.0),
            take_profit_ratio: params.take_profit_pct / 100.0,
        }
    }

    /// Loosen the stop-loss so it sits outside normal ATR-sized noise.
    ///
    /// `atr_pct` is recent ATR as a percentage of price (the volatility
    /// hint carried on [`PositionEvent`]); `atr_multiplier` is how many
    /// ATRs the optimizer places its trailing stop (reused here so the
    /// band matches the strategy's own notion of "normal" movement). If
    /// the ATR band (`atr_multiplier * atr_pct`) is wider than the
    /// configured stop, the stop is widened to match — otherwise the
    /// configured stop is already conservative and is kept.
    ///
    /// Take-profit is intentionally left untouched: it's a target, not
    /// noise protection. Non-positive inputs are a no-op, so callers can
    /// pass values straight through without pre-checking.
    pub fn widen_for_volatility(self, atr_pct: f64, atr_multiplier: f64) -> Self {
        // Guard non-positive / non-finite inputs (NaN fails `is_finite`),
        // so callers can pass producer-supplied values straight through.
        if !atr_pct.is_finite() || atr_pct <= 0.0 || atr_multiplier <= 0.0 {
            return self;
        }
        // ATR band as a positive ratio of notional, then signed negative
        // to match `stop_loss_ratio`'s convention.
        let atr_floor = -(atr_multiplier * atr_pct / 100.0);
        Self {
            // Both negative; the more-negative (wider) stop wins.
            stop_loss_ratio: self.stop_loss_ratio.min(atr_floor),
            ..self
        }
    }

    /// Tighten the stop toward break-even in proportion to elevated fear.
    ///
    /// The inverse of [`widen_for_volatility`]: under amygdala stress we
    /// want a losing position on a shorter leash. `fear` is expected in
    /// the elevated band `[FEAR_ELEVATED_LEVEL, FEAR_EXIT_LEVEL)`; the
    /// stop magnitude scales linearly from full width (at the bottom of
    /// the band) down to [`STOP_TIGHTEN_FLOOR`] of its width (at the top).
    /// It never reaches zero, so a position isn't exited on the first
    /// tick of an adverse move. Fear below the elevated band is a no-op.
    pub fn tighten_stop_for_fear(self, fear: f64) -> Self {
        if !fear.is_finite() || fear < FEAR_ELEVATED_LEVEL {
            return self;
        }
        let span = FEAR_EXIT_LEVEL - FEAR_ELEVATED_LEVEL;
        let t = ((fear - FEAR_ELEVATED_LEVEL) / span).clamp(0.0, 1.0);
        let factor = 1.0 - t * (1.0 - STOP_TIGHTEN_FLOOR);
        Self {
            stop_loss_ratio: self.stop_loss_ratio * factor,
            ..self
        }
    }
}

/// Fear level (inclusive) at or above which guidance exits outright,
/// regardless of P&L — the amygdala equivalent of a crisis regime.
pub const FEAR_EXIT_LEVEL: f64 = 0.8;

/// Fear level (inclusive) at or above which guidance escalates: banks
/// open profit (Reduce) or tightens the stop on a losing position.
pub const FEAR_ELEVATED_LEVEL: f64 = 0.5;

/// Floor for the stop-tightening factor at the top of the elevated band
/// (just below [`FEAR_EXIT_LEVEL`]). `0.25` ⇒ the stop tightens to a
/// quarter of its configured width but never to break-even.
const STOP_TIGHTEN_FLOOR: f64 = 0.25;

/// Compute advisory guidance for an open position.
///
/// Rules in priority order:
/// 1. Crisis-flavoured regime ⇒ [`Exit`](GuidanceAction::Exit).
/// 2. High amygdala fear (≥ [`FEAR_EXIT_LEVEL`]) ⇒ [`Exit`](GuidanceAction::Exit).
/// 3. Elevated fear (≥ [`FEAR_ELEVATED_LEVEL`]): if in profit ⇒
///    [`Reduce`](GuidanceAction::Reduce) (bank it early); if at a loss ⇒
///    tighten the stop toward break-even before the P&L checks below.
/// 4. Unrealized loss ≤ `thresholds.stop_loss_ratio` of notional ⇒
///    [`Exit`](GuidanceAction::Exit).
/// 5. Unrealized gain ≥ `thresholds.take_profit_ratio` of notional ⇒
///    [`Reduce`](GuidanceAction::Reduce).
/// 6. Otherwise ⇒ [`Hold`](GuidanceAction::Hold).
///
/// `fear` is the latest amygdala threat level (`0.0..=1.0`), or `None`
/// when no producer has reported one. `notional = entry_price * qty`
/// (sign-agnostic; works for both Buy and Sell positions because
/// `pnl_unrealized` is supplied with sign already).
pub fn compute_guidance(
    event: &PositionEvent,
    regime: Option<&str>,
    thresholds: GuidanceThresholds,
    fear: Option<f64>,
) -> Guidance {
    if let Some(label) = regime
        && is_crisis_regime(label)
    {
        return Guidance::exit(format!("regime: {label}"));
    }

    // Amygdala fear escalation (graduated: bank winners, tighten losers).
    let mut thresholds = thresholds;
    if let Some(fear) = fear.filter(|f| f.is_finite()) {
        if fear >= FEAR_EXIT_LEVEL {
            return Guidance::exit(format!("fear {fear:.2} ≥ {FEAR_EXIT_LEVEL}"));
        }
        if fear >= FEAR_ELEVATED_LEVEL {
            if event.pnl_unrealized > 0.0 {
                return Guidance::reduce(format!("fear {fear:.2}: banking open profit"));
            }
            // At a loss or flat: shorten the leash before the P&L checks.
            thresholds = thresholds.tighten_stop_for_fear(fear);
        }
    }

    if let Some(ratio) = event.pnl_ratio() {
        if ratio <= thresholds.stop_loss_ratio {
            let pct = ratio * 100.0;
            return Guidance::exit(format!("stop loss: {pct:.2}% of notional"));
        }
        if ratio >= thresholds.take_profit_ratio {
            let pct = ratio * 100.0;
            return Guidance::reduce(format!("take profit: {pct:.2}% of notional"));
        }
    }

    Guidance::hold("within bounds")
}

/// Extract the base asset from a position symbol — `"BTC-USD"` → `"BTC"`,
/// `"ETH/USDT"` → `"ETH"`, `"BTC"` → `"BTC"`. Used to look up per-asset
/// [`OptimizedParams`] from a [`ParamManager`](crate::optimized_params::ParamManager).
pub fn base_asset(symbol: &str) -> &str {
    symbol
        .split(['-', '/'])
        .next()
        .filter(|s| !s.is_empty())
        .unwrap_or(symbol)
}

fn is_crisis_regime(label: &str) -> bool {
    let lower = label.to_ascii_lowercase();
    ["crisis", "panic", "flash_crash", "shock"]
        .iter()
        .any(|needle| lower.contains(needle))
}

/// Rolling per-position state, accumulated across the repeated snapshots a
/// producer pushes for the same `position_id`. Lets guidance depend on a
/// position's *history* (peak profit, prior advice) rather than scoring each
/// snapshot in isolation.
#[derive(Debug, Clone)]
pub struct PositionState {
    /// Instant the first snapshot for this position arrived.
    pub first_seen: Instant,
    /// Instant the most recent snapshot arrived (drives TTL eviction).
    pub last_seen: Instant,
    /// Number of snapshots observed for this position.
    pub samples: u64,
    /// Highest unrealized-P&L ratio (`pnl / notional`) seen so far.
    pub peak_pnl_ratio: f64,
    /// Most recent guidance action issued for this position.
    pub last_action: GuidanceAction,
}

impl PositionState {
    /// How long this position has been reported.
    pub fn time_in_position(&self) -> Duration {
        self.last_seen.duration_since(self.first_seen)
    }
}

/// Tunables for the stateful guidance layer in [`PositionTracker`].
#[derive(Debug, Clone, Copy)]
pub struct TrailingConfig {
    /// Peak gain (ratio of notional) a position must reach before the
    /// trailing give-back rule arms. Peaks below this are too small to act on.
    pub arm_ratio: f64,
    /// Fraction of the peak gain that, once surrendered, triggers a `Reduce`
    /// to lock in what's left. `0.5` ⇒ "gave back half the peak".
    pub giveback_frac: f64,
    /// Evict positions not seen within this window (presumed closed/stale).
    pub ttl: Duration,
    /// Hard cap on tracked positions, so a misbehaving producer can't grow
    /// the map without bound.
    pub max_entries: usize,
}

impl Default for TrailingConfig {
    fn default() -> Self {
        Self {
            arm_ratio: 0.03,
            giveback_frac: 0.5,
            ttl: Duration::from_secs(3600),
            max_entries: 10_000,
        }
    }
}

/// Stateful refinement layer on top of [`compute_guidance`].
///
/// [`compute_guidance`] is pure and scores a single snapshot. The tracker
/// remembers each position (keyed by `position_id`) and adds two rules that
/// require history:
///
/// 1. **Trailing give-back** — once a position has been a meaningful winner
///    (peak ≥ [`TrailingConfig::arm_ratio`]) and then surrenders enough of
///    that peak ([`TrailingConfig::giveback_frac`]), upgrade a `Hold` to
///    `Reduce` to bank the remaining profit. The stateless rule can't see the
///    prior peak, so it would just `Hold`.
/// 2. **Sticky exit** — once we've advised `Exit`, a later snapshot that
///    merely drifts back inside the bands keeps `Exit` rather than flip-
///    flopping to `Hold` on a one-tick bounce.
///
/// Snapshots without a `position_id` are not tracked; their guidance passes
/// through unchanged (fully backward compatible).
pub struct PositionTracker {
    states: RwLock<HashMap<String, PositionState>>,
    config: TrailingConfig,
}

impl Default for PositionTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl PositionTracker {
    /// New tracker with default [`TrailingConfig`].
    pub fn new() -> Self {
        Self::with_config(TrailingConfig::default())
    }

    /// New tracker with explicit config.
    pub fn with_config(config: TrailingConfig) -> Self {
        Self {
            states: RwLock::new(HashMap::new()),
            config,
        }
    }

    /// Number of positions currently tracked (after pruning). For metrics/tests.
    pub async fn tracked(&self) -> usize {
        self.states.read().await.len()
    }

    /// Remove and return a position's accumulated state. Call this when the
    /// position closes so its guidance history can be joined with the realized
    /// outcome (see [`PositionOutcome::from_close`]). Returns `None` if the
    /// position was never tracked.
    pub async fn finalize(&self, position_id: &str) -> Option<PositionState> {
        self.states.write().await.remove(position_id)
    }

    /// Refine stateless `base` guidance using this position's history, record
    /// the updated state, and return the (possibly upgraded) guidance.
    ///
    /// Positions without a `position_id` are returned unchanged and not
    /// tracked. The critical section is pure (no `.await` while the lock is
    /// held), so this stays cheap under load.
    pub async fn observe(&self, event: &PositionEvent, base: Guidance) -> Guidance {
        let Some(key) = event.position_id.clone() else {
            return base;
        };
        let ratio = event.pnl_ratio().unwrap_or(0.0);
        let now = Instant::now();

        let mut states = self.states.write().await;

        // Best-effort TTL prune, then a hard cap (evict the least-recently
        // seen) so the map can't grow without bound on a long-lived process.
        states.retain(|_, s| now.duration_since(s.last_seen) <= self.config.ttl);
        if !states.contains_key(&key)
            && states.len() >= self.config.max_entries
            && let Some(oldest) = states
                .iter()
                .min_by_key(|(_, s)| s.last_seen)
                .map(|(k, _)| k.clone())
        {
            states.remove(&oldest);
        }

        let entry = states.entry(key).or_insert_with(|| PositionState {
            first_seen: now,
            last_seen: now,
            samples: 0,
            peak_pnl_ratio: ratio,
            last_action: GuidanceAction::Hold,
        });
        let prior_action = entry.last_action;
        entry.last_seen = now;
        entry.samples += 1;
        entry.peak_pnl_ratio = entry.peak_pnl_ratio.max(ratio);
        let peak = entry.peak_pnl_ratio;

        let mut action = base.action;
        let mut reason = base.reason;

        // (1) Trailing give-back: only ever *upgrades* a Hold while still in
        // profit. A loss is the stop-loss's job, and we never downgrade a
        // Reduce/Exit.
        if action == GuidanceAction::Hold
            && ratio > 0.0
            && peak >= self.config.arm_ratio
            && ratio <= peak * (1.0 - self.config.giveback_frac)
        {
            action = GuidanceAction::Reduce;
            reason = format!(
                "trailing: gave back to {:.2}% from {:.2}% peak",
                ratio * 100.0,
                peak * 100.0
            );
        }

        // (2) Sticky exit: don't rescind a prior Exit on a small bounce.
        if prior_action == GuidanceAction::Exit && action != GuidanceAction::Exit {
            action = GuidanceAction::Exit;
            reason = "prior exit still standing".to_string();
        }

        entry.last_action = action;
        Guidance { action, reason }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> PositionEvent {
        PositionEvent {
            symbol: "BTC-USD".to_string(),
            side: Side::Buy,
            qty: 0.5,
            entry_price: 60_000.0,
            current_price: 61_000.0,
            pnl_unrealized: 500.0,
            position_id: Some("pos-1".to_string()),
            session_id: Some("sess-1".to_string()),
            atr_pct: None,
        }
    }

    #[test]
    fn validate_accepts_well_formed_event() {
        assert!(sample().validate().is_ok());
    }

    #[test]
    fn validate_rejects_empty_symbol() {
        let mut e = sample();
        e.symbol.clear();
        assert_eq!(e.validate(), Err("symbol is empty"));
    }

    #[test]
    fn validate_rejects_non_positive_qty() {
        let mut e = sample();
        e.qty = 0.0;
        assert!(e.validate().is_err());
        e.qty = -1.0;
        assert!(e.validate().is_err());
    }

    #[test]
    fn validate_rejects_non_finite_prices() {
        let mut e = sample();
        e.entry_price = f64::NAN;
        assert!(e.validate().is_err());

        let mut e = sample();
        e.current_price = f64::INFINITY;
        assert!(e.validate().is_err());

        let mut e = sample();
        e.pnl_unrealized = f64::NAN;
        assert!(e.validate().is_err());
    }

    #[test]
    fn round_trips_through_json_with_optional_fields_omitted() {
        let e = PositionEvent {
            position_id: None,
            session_id: None,
            ..sample()
        };
        let json = serde_json::to_string(&e).unwrap();
        assert!(!json.contains("position_id"));
        assert!(!json.contains("session_id"));
        let back: PositionEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back.symbol, e.symbol);
        assert!(back.position_id.is_none());
    }

    #[test]
    fn deserializes_minimal_payload() {
        let json = r#"{
            "symbol": "ETH-USD",
            "side": "Sell",
            "qty": 2.0,
            "entry_price": 3000.0,
            "current_price": 2950.0,
            "pnl_unrealized": 100.0
        }"#;
        let e: PositionEvent = serde_json::from_str(json).unwrap();
        assert_eq!(e.symbol, "ETH-USD");
        assert_eq!(e.side, Side::Sell);
        assert!(e.position_id.is_none());
    }

    // ── Guidance ─────────────────────────────────────────────────────

    fn default_thresholds() -> GuidanceThresholds {
        GuidanceThresholds::default()
    }

    #[test]
    fn guidance_holds_when_within_bounds() {
        // 0.5 BTC at 60_000 = 30_000 notional. +100 pnl = 0.33% — below take-profit.
        let mut e = sample();
        e.pnl_unrealized = 100.0;
        let g = compute_guidance(&e, None, default_thresholds(), None);
        assert_eq!(g.action, GuidanceAction::Hold);
    }

    #[test]
    fn guidance_exits_on_stop_loss_breach() {
        // Notional 30_000; -2% = -600. -700 trips the stop.
        let mut e = sample();
        e.pnl_unrealized = -700.0;
        let g = compute_guidance(&e, None, default_thresholds(), None);
        assert_eq!(g.action, GuidanceAction::Exit);
        assert!(g.reason.contains("stop loss"));
    }

    #[test]
    fn guidance_reduces_on_take_profit() {
        // Notional 30_000; +5% = +1500. +2000 trips the take-profit.
        let mut e = sample();
        e.pnl_unrealized = 2_000.0;
        let g = compute_guidance(&e, None, default_thresholds(), None);
        assert_eq!(g.action, GuidanceAction::Reduce);
        assert!(g.reason.contains("take profit"));
    }

    #[test]
    fn guidance_exits_on_crisis_regime_regardless_of_pnl() {
        // Even with healthy pnl, a crisis regime triggers exit.
        let mut e = sample();
        e.pnl_unrealized = 100.0;
        let g = compute_guidance(&e, Some("crisis_volatility_spike"), default_thresholds(), None);
        assert_eq!(g.action, GuidanceAction::Exit);
        assert!(g.reason.contains("regime"));
    }

    #[test]
    fn guidance_crisis_detection_is_case_insensitive_and_substring() {
        let e = sample();
        for label in ["PANIC", "Flash_Crash detected", "shockwave"] {
            assert_eq!(
                compute_guidance(&e, Some(label), default_thresholds(), None).action,
                GuidanceAction::Exit,
                "label {label:?} should trigger exit"
            );
        }
    }

    #[test]
    fn guidance_ignores_unknown_regime_labels() {
        let e = sample();
        // "bullish_trend" isn't a crisis label, so guidance is pnl-driven.
        assert_eq!(
            compute_guidance(&e, Some("bullish_trend"), default_thresholds(), None).action,
            GuidanceAction::Hold
        );
    }

    #[test]
    fn guidance_action_serializes_lowercase() {
        let g = Guidance::hold("ok");
        let json = serde_json::to_string(&g).unwrap();
        assert!(json.contains("\"action\":\"hold\""));
    }

    // ── GuidanceThresholds ───────────────────────────────────────────

    #[test]
    fn thresholds_from_optimized_params_overrides_both_ratios() {
        let params = OptimizedParams {
            take_profit_pct: 8.0,
            stop_loss_pct: 3.0, // optimizer tuned to tighter stop
            ..OptimizedParams::new("BTC")
        };
        let t = GuidanceThresholds::from_optimized_params(&params);
        assert!((t.take_profit_ratio - 0.08).abs() < 1e-9);
        // stop_loss_pct is positive; ratio must be negative.
        assert!((t.stop_loss_ratio - (-0.03)).abs() < 1e-9);
    }

    #[test]
    fn thresholds_default_stop_loss_matches_optimized_params_default() {
        // Ensure serde default (2.0%) and GuidanceThresholds::default (-0.02)
        // stay in sync — a divergence would silently change live behaviour.
        let params = OptimizedParams::default();
        let t = GuidanceThresholds::from_optimized_params(&params);
        assert_eq!(t.stop_loss_ratio, GuidanceThresholds::default().stop_loss_ratio);
        assert_eq!(t.take_profit_ratio, GuidanceThresholds::default().take_profit_ratio);
    }

    // ── widen_for_volatility ─────────────────────────────────────────

    #[test]
    fn widen_for_volatility_loosens_stop_when_atr_band_is_wider() {
        // Default stop -2%. ATR 1.5% × multiplier 2.0 = 3% band → wider,
        // so the stop should loosen to -3%. Take-profit untouched.
        let t = GuidanceThresholds::default().widen_for_volatility(1.5, 2.0);
        assert!((t.stop_loss_ratio - (-0.03)).abs() < 1e-9);
        assert_eq!(t.take_profit_ratio, GuidanceThresholds::default().take_profit_ratio);
    }

    #[test]
    fn widen_for_volatility_keeps_stop_when_already_wider() {
        // Configured stop -5%. ATR band 0.5% × 2.0 = 1% → narrower, so the
        // already-conservative configured stop is kept.
        let base = GuidanceThresholds {
            stop_loss_ratio: -0.05,
            ..GuidanceThresholds::default()
        };
        let t = base.widen_for_volatility(0.5, 2.0);
        assert!((t.stop_loss_ratio - (-0.05)).abs() < 1e-9);
    }

    #[test]
    fn widen_for_volatility_is_noop_for_nonpositive_inputs() {
        let base = GuidanceThresholds::default();
        assert_eq!(base.widen_for_volatility(0.0, 2.0), base);
        assert_eq!(base.widen_for_volatility(1.5, 0.0), base);
        assert_eq!(base.widen_for_volatility(-1.0, 2.0), base);
    }

    #[test]
    fn guidance_volatility_widened_stop_avoids_noise_exit() {
        // Notional 30_000. -800 pnl ≈ -2.67% → trips the default -2% stop.
        let mut e = sample();
        e.pnl_unrealized = -800.0;
        assert_eq!(
            compute_guidance(&e, None, GuidanceThresholds::default(), None).action,
            GuidanceAction::Exit
        );
        // But with ATR 2% × multiplier 2.0 = 4% band, the stop widens to -4%
        // and the same -2.67% move is now treated as noise → hold.
        let widened = GuidanceThresholds::default().widen_for_volatility(2.0, 2.0);
        assert_eq!(
            compute_guidance(&e, None, widened, None).action,
            GuidanceAction::Hold
        );
    }

    #[test]
    fn validate_rejects_negative_atr_pct() {
        let mut e = sample();
        e.atr_pct = Some(-0.5);
        assert!(e.validate().is_err());
        e.atr_pct = Some(f64::NAN);
        assert!(e.validate().is_err());
        e.atr_pct = Some(0.0); // zero is allowed (flat / unknown)
        assert!(e.validate().is_ok());
    }

    #[test]
    fn guidance_take_profit_uses_supplied_threshold() {
        // Bump take-profit to 10%. +1500 pnl on 30_000 notional = 5% — below the
        // tuned threshold, so guidance should NOT reduce.
        let mut e = sample();
        e.pnl_unrealized = 1_500.0;
        let tighter = GuidanceThresholds {
            take_profit_ratio: 0.10,
            ..GuidanceThresholds::default()
        };
        assert_eq!(
            compute_guidance(&e, None, tighter, None).action,
            GuidanceAction::Hold
        );
    }

    #[test]
    fn guidance_stop_loss_uses_supplied_threshold() {
        // Tighten stop to 1%. -200 pnl on 30_000 notional ≈ -0.67% —
        // below the default -2% but above the tighter -1%, so it should hold.
        let mut e = sample();
        e.pnl_unrealized = -200.0;
        let tighter = GuidanceThresholds {
            stop_loss_ratio: -0.01,
            ..GuidanceThresholds::default()
        };
        assert_eq!(
            compute_guidance(&e, None, tighter, None).action,
            GuidanceAction::Hold,
            "-0.67% loss should hold under a 1% stop threshold"
        );
        // But -350 pnl ≈ -1.17% should now trip the tighter stop.
        let mut e2 = sample();
        e2.pnl_unrealized = -350.0;
        assert_eq!(
            compute_guidance(&e2, None, tighter, None).action,
            GuidanceAction::Exit,
            "-1.17% loss should exit under a 1% stop threshold"
        );
    }

    // ── Amygdala fear escalation ─────────────────────────────────────

    #[test]
    fn guidance_high_fear_exits_regardless_of_pnl() {
        // Healthy +100 pnl, but fear ≥ 0.8 forces an exit.
        let mut e = sample();
        e.pnl_unrealized = 100.0;
        let g = compute_guidance(&e, None, default_thresholds(), Some(0.85));
        assert_eq!(g.action, GuidanceAction::Exit);
        assert!(g.reason.contains("fear"), "reason was: {}", g.reason);
    }

    #[test]
    fn guidance_elevated_fear_banks_open_profit() {
        // In profit but below the take-profit threshold; elevated fear
        // (0.6) banks it early via Reduce rather than waiting.
        let mut e = sample();
        e.pnl_unrealized = 100.0; // 0.33% of notional — under the 5% TP
        let g = compute_guidance(&e, None, default_thresholds(), Some(0.6));
        assert_eq!(g.action, GuidanceAction::Reduce);
        assert!(g.reason.contains("banking"), "reason was: {}", g.reason);
    }

    #[test]
    fn guidance_elevated_fear_tightens_stop_on_a_loser() {
        // -300 pnl ≈ -1% of notional — inside the default -2% stop, so a
        // calm market holds. At fear 0.8-ε the stop tightens to ~25% of
        // its width (≈ -0.5%), so the same -1% loss now exits.
        let mut e = sample();
        e.pnl_unrealized = -300.0;
        assert_eq!(
            compute_guidance(&e, None, default_thresholds(), None).action,
            GuidanceAction::Hold,
            "-1% loss holds with no fear"
        );
        let g = compute_guidance(&e, None, default_thresholds(), Some(0.79));
        assert_eq!(
            g.action,
            GuidanceAction::Exit,
            "tightened stop under high-elevated fear should exit a -1% loser"
        );
    }

    #[test]
    fn guidance_low_fear_is_inert() {
        // Fear below the elevated band changes nothing.
        let mut e = sample();
        e.pnl_unrealized = 100.0;
        assert_eq!(
            compute_guidance(&e, None, default_thresholds(), Some(0.3)).action,
            GuidanceAction::Hold
        );
    }

    #[test]
    fn guidance_crisis_regime_outranks_fear_reduce() {
        // A crisis regime exits even when fear would only Reduce a winner.
        let mut e = sample();
        e.pnl_unrealized = 100.0;
        let g = compute_guidance(&e, Some("crisis"), default_thresholds(), Some(0.6));
        assert_eq!(g.action, GuidanceAction::Exit);
        assert!(g.reason.contains("regime"), "reason was: {}", g.reason);
    }

    #[test]
    fn tighten_stop_for_fear_scales_within_band() {
        let base = GuidanceThresholds::default(); // stop -0.02
        // Bottom of band: no change.
        assert!(
            (base.tighten_stop_for_fear(FEAR_ELEVATED_LEVEL).stop_loss_ratio - (-0.02)).abs()
                < 1e-9
        );
        // Top of band (→ exit level): tightened to STOP_TIGHTEN_FLOOR (0.25) of width.
        let top = base.tighten_stop_for_fear(FEAR_EXIT_LEVEL - 1e-9).stop_loss_ratio;
        assert!((top - (-0.005)).abs() < 1e-4, "got {top}");
        // Below band: inert.
        assert_eq!(base.tighten_stop_for_fear(0.2), base);
    }

    // ── base_asset ───────────────────────────────────────────────────

    #[test]
    fn base_asset_strips_quote_currency_suffix() {
        assert_eq!(base_asset("BTC-USD"), "BTC");
        assert_eq!(base_asset("ETH/USDT"), "ETH");
        assert_eq!(base_asset("SOL"), "SOL");
        assert_eq!(base_asset(""), "");
    }

    // ── PositionTracker (stateful guidance) ──────────────────────────

    /// Event with a fixed 30_000 notional (entry 60_000 × qty 0.5), so
    /// `pnl_ratio() == pnl / 30_000`.
    fn ev(position_id: Option<&str>, pnl: f64) -> PositionEvent {
        PositionEvent {
            symbol: "BTC-USD".to_string(),
            side: Side::Buy,
            qty: 0.5,
            entry_price: 60_000.0,
            current_price: 60_000.0,
            pnl_unrealized: pnl,
            position_id: position_id.map(String::from),
            session_id: None,
            atr_pct: None,
        }
    }

    #[test]
    fn pnl_ratio_uses_entry_notional() {
        assert!((ev(None, 1500.0).pnl_ratio().unwrap() - 0.05).abs() < 1e-9);
        // Non-positive notional → None (guards a divide-by-zero downstream).
        let mut e = ev(None, 100.0);
        e.entry_price = 0.0;
        assert!(e.pnl_ratio().is_none());
    }

    #[tokio::test]
    async fn tracker_passes_through_untracked_when_no_position_id() {
        let tracker = PositionTracker::new();
        let g = tracker
            .observe(&ev(None, 600.0), Guidance::hold("within bounds"))
            .await;
        assert_eq!(g.action, GuidanceAction::Hold);
        assert_eq!(tracker.tracked().await, 0);
    }

    #[tokio::test]
    async fn tracker_trailing_reduces_after_giveback() {
        let tracker = PositionTracker::new();
        // Peaks at +10% (a legit take-profit Reduce), establishing the peak.
        let g1 = tracker
            .observe(&ev(Some("p1"), 3000.0), Guidance::reduce("take profit"))
            .await;
        assert_eq!(g1.action, GuidanceAction::Reduce);
        // Fades to +4%: the stateless rule would Hold, but half of the 10%
        // peak is gone → trailing upgrades to Reduce.
        let g2 = tracker
            .observe(&ev(Some("p1"), 1200.0), Guidance::hold("within bounds"))
            .await;
        assert_eq!(g2.action, GuidanceAction::Reduce);
        assert!(g2.reason.contains("trailing"), "reason was: {}", g2.reason);
    }

    #[tokio::test]
    async fn tracker_trailing_inert_when_peak_below_arm() {
        let tracker = PositionTracker::new();
        // Peak only +2% — below the 3% arm threshold, so give-back is ignored.
        tracker
            .observe(&ev(Some("p2"), 600.0), Guidance::hold("within bounds"))
            .await;
        let g = tracker
            .observe(&ev(Some("p2"), 150.0), Guidance::hold("within bounds"))
            .await;
        assert_eq!(g.action, GuidanceAction::Hold);
    }

    #[tokio::test]
    async fn tracker_sticky_exit_survives_a_bounce() {
        let tracker = PositionTracker::new();
        // First snapshot exits (e.g. crisis regime); small pnl so trailing
        // never arms and can't be the cause of a later non-Hold.
        let g1 = tracker
            .observe(&ev(Some("p3"), 100.0), Guidance::exit("regime: crisis"))
            .await;
        assert_eq!(g1.action, GuidanceAction::Exit);
        // Price drifts back inside the bands → stateless Hold, but the prior
        // Exit still stands.
        let g2 = tracker
            .observe(&ev(Some("p3"), 100.0), Guidance::hold("within bounds"))
            .await;
        assert_eq!(g2.action, GuidanceAction::Exit);
        assert!(g2.reason.contains("prior exit"), "reason was: {}", g2.reason);
    }

    #[tokio::test]
    async fn tracker_caps_tracked_positions() {
        let tracker = PositionTracker::with_config(TrailingConfig {
            max_entries: 2,
            ..TrailingConfig::default()
        });
        for id in ["a", "b", "c"] {
            tracker
                .observe(&ev(Some(id), 600.0), Guidance::hold("within bounds"))
                .await;
        }
        assert_eq!(tracker.tracked().await, 2, "oldest entry should be evicted");
    }

    // ── PositionClose / PositionOutcome (outcome capture) ────────────

    fn close_ev(position_id: Option<&str>, pnl_realized: f64) -> PositionClose {
        PositionClose {
            symbol: "BTC-USD".to_string(),
            side: Side::Buy,
            qty: 0.5,
            entry_price: 60_000.0,
            exit_price: 60_500.0,
            pnl_realized,
            rr_ratio: None,
            strategy: None,
            position_id: position_id.map(String::from),
            session_id: None,
        }
    }

    #[test]
    fn position_close_validate_rejects_bad_input() {
        assert!(close_ev(None, 100.0).validate().is_ok());

        let mut e = close_ev(None, 100.0);
        e.symbol.clear();
        assert!(e.validate().is_err());

        let mut e = close_ev(None, 100.0);
        e.qty = -1.0;
        assert!(e.validate().is_err());

        let mut e = close_ev(None, 100.0);
        e.exit_price = 0.0;
        assert!(e.validate().is_err());

        let e = close_ev(None, f64::NAN);
        assert!(e.validate().is_err());
    }

    #[test]
    fn outcome_from_close_without_state_is_untracked() {
        // 30_000 notional; +1500 realized = +5% → Win, but no tracker state.
        let o = PositionOutcome::from_close(&close_ev(None, 1500.0), None);
        assert_eq!(o.result, OutcomeResult::Win);
        assert!((o.realized_ratio - 0.05).abs() < 1e-9);
        assert_eq!(o.samples, 0);
        assert!(o.peak_pnl_ratio.is_none());
        assert!(o.last_guidance.is_none());
        assert!(o.time_in_position_secs.is_none());

        // Sign classification.
        assert_eq!(
            PositionOutcome::from_close(&close_ev(None, -600.0), None).result,
            OutcomeResult::Loss
        );
        assert_eq!(
            PositionOutcome::from_close(&close_ev(None, 0.0), None).result,
            OutcomeResult::Breakeven
        );
    }

    #[test]
    fn outcome_from_close_joins_tracker_state() {
        let now = Instant::now();
        let state = PositionState {
            first_seen: now,
            last_seen: now,
            samples: 3,
            peak_pnl_ratio: 0.08,
            last_action: GuidanceAction::Reduce,
        };
        let o = PositionOutcome::from_close(&close_ev(Some("p"), 300.0), Some(&state));
        assert_eq!(o.samples, 3);
        assert_eq!(o.peak_pnl_ratio, Some(0.08));
        assert_eq!(o.last_guidance, Some(GuidanceAction::Reduce));
        assert!(o.time_in_position_secs.is_some());
        assert_eq!(o.result, OutcomeResult::Win); // +300 / 30_000 = +1%
    }

    #[test]
    fn outcome_carries_strategy_and_rr_for_affinity() {
        let mut close = close_ev(Some("p"), 900.0);
        close.strategy = Some("ema_cross".to_string());
        close.rr_ratio = Some(2.5);
        let o = PositionOutcome::from_close(&close, None);
        assert_eq!(o.strategy.as_deref(), Some("ema_cross"));
        assert_eq!(o.rr_ratio, Some(2.5));
        assert!(o.is_winner(), "+900 realized is a win");
        // Breakeven and losses are not winners.
        assert!(!PositionOutcome::from_close(&close_ev(None, 0.0), None).is_winner());
        assert!(!PositionOutcome::from_close(&close_ev(None, -50.0), None).is_winner());
    }

    #[tokio::test]
    async fn tracker_finalize_removes_and_returns_state() {
        let tracker = PositionTracker::new();
        tracker
            .observe(&ev(Some("p"), 3000.0), Guidance::reduce("take profit"))
            .await;
        let state = tracker.finalize("p").await.expect("position was tracked");
        assert_eq!(state.samples, 1);
        assert!((state.peak_pnl_ratio - 0.10).abs() < 1e-9);
        assert_eq!(state.last_action, GuidanceAction::Reduce);
        // Finalize consumes the entry.
        assert_eq!(tracker.tracked().await, 0);
        assert!(tracker.finalize("p").await.is_none());
    }
}
