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
        Ok(())
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
    /// Only `take_profit_ratio` is currently learnable —
    /// [`OptimizedParams`] doesn't carry an explicit stop-loss field
    /// (see TODO note in the position-feedback section of `TODO.md`),
    /// so the stop ratio is kept at the conservative default until
    /// the optimizer schema grows one.
    pub fn from_optimized_params(params: &OptimizedParams) -> Self {
        Self {
            take_profit_ratio: params.take_profit_pct / 100.0,
            ..Self::default()
        }
    }
}

/// Compute advisory guidance for an open position.
///
/// Rules in priority order:
/// 1. Crisis-flavoured regime ⇒ [`Exit`](GuidanceAction::Exit).
/// 2. Unrealized loss ≤ `thresholds.stop_loss_ratio` of notional ⇒
///    [`Exit`](GuidanceAction::Exit).
/// 3. Unrealized gain ≥ `thresholds.take_profit_ratio` of notional ⇒
///    [`Reduce`](GuidanceAction::Reduce).
/// 4. Otherwise ⇒ [`Hold`](GuidanceAction::Hold).
///
/// `notional = entry_price * qty` (sign-agnostic; works for both Buy and
/// Sell positions because `pnl_unrealized` is supplied with sign already).
pub fn compute_guidance(
    event: &PositionEvent,
    regime: Option<&str>,
    thresholds: GuidanceThresholds,
) -> Guidance {
    if let Some(label) = regime
        && is_crisis_regime(label)
    {
        return Guidance::exit(format!("regime: {label}"));
    }

    let notional = event.entry_price * event.qty;
    if notional > 0.0 {
        let ratio = event.pnl_unrealized / notional;
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
        let g = compute_guidance(&e, None, default_thresholds());
        assert_eq!(g.action, GuidanceAction::Hold);
    }

    #[test]
    fn guidance_exits_on_stop_loss_breach() {
        // Notional 30_000; -2% = -600. -700 trips the stop.
        let mut e = sample();
        e.pnl_unrealized = -700.0;
        let g = compute_guidance(&e, None, default_thresholds());
        assert_eq!(g.action, GuidanceAction::Exit);
        assert!(g.reason.contains("stop loss"));
    }

    #[test]
    fn guidance_reduces_on_take_profit() {
        // Notional 30_000; +5% = +1500. +2000 trips the take-profit.
        let mut e = sample();
        e.pnl_unrealized = 2_000.0;
        let g = compute_guidance(&e, None, default_thresholds());
        assert_eq!(g.action, GuidanceAction::Reduce);
        assert!(g.reason.contains("take profit"));
    }

    #[test]
    fn guidance_exits_on_crisis_regime_regardless_of_pnl() {
        // Even with healthy pnl, a crisis regime triggers exit.
        let mut e = sample();
        e.pnl_unrealized = 100.0;
        let g = compute_guidance(&e, Some("crisis_volatility_spike"), default_thresholds());
        assert_eq!(g.action, GuidanceAction::Exit);
        assert!(g.reason.contains("regime"));
    }

    #[test]
    fn guidance_crisis_detection_is_case_insensitive_and_substring() {
        let e = sample();
        for label in ["PANIC", "Flash_Crash detected", "shockwave"] {
            assert_eq!(
                compute_guidance(&e, Some(label), default_thresholds()).action,
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
            compute_guidance(&e, Some("bullish_trend"), default_thresholds()).action,
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
    fn thresholds_from_optimized_params_overrides_take_profit_only() {
        let mut params = OptimizedParams::default(); // take_profit_pct = 5.0
        params.take_profit_pct = 8.0; // optimizer tuned to 8%
        let t = GuidanceThresholds::from_optimized_params(&params);
        assert!((t.take_profit_ratio - 0.08).abs() < 1e-9);
        // stop_loss_ratio inherits the default until OptimizedParams grows one.
        assert_eq!(t.stop_loss_ratio, GuidanceThresholds::default().stop_loss_ratio);
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
            compute_guidance(&e, None, tighter).action,
            GuidanceAction::Hold
        );
    }

    // ── base_asset ───────────────────────────────────────────────────

    #[test]
    fn base_asset_strips_quote_currency_suffix() {
        assert_eq!(base_asset("BTC-USD"), "BTC");
        assert_eq!(base_asset("ETH/USDT"), "ETH");
        assert_eq!(base_asset("SOL"), "SOL");
        assert_eq!(base_asset(""), "");
    }
}
