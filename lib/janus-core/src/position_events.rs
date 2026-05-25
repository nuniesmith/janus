//! Position event ingress (JFLOW-C foundation).
//!
//! Receives live position snapshots from the execution side (Ruby / fks). This
//! first slice defines the wire shape and a validator. Receiving HTTP handler
//! lives in `janus-api`; guidance computation, regime-aware exits, and
//! persistence into `janus_memories` are follow-ups.

use crate::market::Side;
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
}
