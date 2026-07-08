//! Inputs to a gate evaluation.

use serde::{Deserialize, Serialize};

/// Proposed trade side — mirrors Ruby's `signal: str` (`"buy"` / `"sell"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Side {
    Buy,
    Sell,
}

impl Side {
    /// The lowercase string form (`"buy"` / `"sell"`), matching the Python signal.
    pub fn as_str(self) -> &'static str {
        match self {
            Side::Buy => "buy",
            Side::Sell => "sell",
        }
    }

    /// Parse a side from a string. Accepts `buy`/`long` and `sell`/`short`
    /// (case-insensitive); returns `None` for anything else.
    pub fn parse(s: &str) -> Option<Side> {
        match s.trim().to_ascii_lowercase().as_str() {
            "buy" | "long" => Some(Side::Buy),
            "sell" | "short" => Some(Side::Sell),
            _ => None,
        }
    }
}

/// A CNN inference vote, as the gate consumes it.
///
/// Mirrors the subset of Ruby's `InferenceResult` the gate touches:
/// `.signal`, `.confidence`, and `.agrees_with(signal)`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CnnVote {
    /// The side the model predicts.
    pub side: Side,
    /// Model confidence in `[0, 1]`.
    pub confidence: f64,
}

impl CnnVote {
    pub fn new(side: Side, confidence: f64) -> Self {
        Self { side, confidence }
    }

    /// `true` when the model's predicted side matches the proposed entry side.
    pub fn agrees_with(&self, side: Side) -> bool {
        self.side == side
    }
}

/// All per-evaluation inputs the gates read.
///
/// Field defaults mirror Ruby's `ctx.get(key, default)` fallbacks exactly, so a
/// partially-populated context behaves identically to the Python gate. In the
/// live loop the trading code fills these from the indicator/risk/CNN state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GateContext {
    /// Result of the risk engine's `can_trade()` check.
    pub can_trade: bool,
    /// Human-readable reason when `can_trade` is false.
    pub risk_reason: String,
    /// Realised volatility percentile in `[0, 1]`.
    pub vol_pct: f64,
    /// Lower bound of the volatility entry band.
    pub vol_entry_min: f64,
    /// Upper bound of the volatility entry band.
    pub vol_entry_max: f64,
    /// Signal-quality score (0–100).
    pub quality: f64,
    /// Minimum acceptable quality score.
    pub qual_min: f64,
    /// Awesome-Oscillator value (sign used for divergence check).
    pub ao: f64,
    /// Whether the AO-divergence gate is active. Kill-switch wired from
    /// `JANUS_GATE_AO_DIVERGENCE` (default enabled); `false` skips the gate.
    pub ao_divergence_enabled: bool,
    /// `true` when this entry is a mean-reversion decision (market regime is
    /// Mean-Reverting, or a mean-reversion strategy). The AO-divergence gate is
    /// trend-following, so it is skipped for these entries — janus's dominant
    /// regime enters *against* AO by design (e.g. RSI-reversal oversold buys).
    pub mean_reversion: bool,
    /// Take-profit target as a fraction (e.g. `0.004` = 0.4%).
    pub tp_pct: f64,
    /// Taker fee as a fraction (per side).
    pub fee_taker: f64,
    /// Assumed slippage as a fraction (per side).
    pub fee_slip: f64,
    /// Whether CNN gating is active for this asset.
    pub cnn_enabled: bool,
    /// CNN vote, if one was produced.
    pub cnn_result: Option<CnnVote>,
    /// Fixed minimum CNN confidence (used when no adaptive threshold is set).
    pub min_confidence: f64,
    /// Assets with currently-open positions (for the correlation gate).
    pub open_assets: Vec<String>,
}

impl Default for GateContext {
    fn default() -> Self {
        // These literals are the Python `ctx.get(key, default)` fallbacks.
        Self {
            can_trade: true,
            risk_reason: "risk limit".to_string(),
            vol_pct: 0.5,
            vol_entry_min: 0.15,
            vol_entry_max: 0.90,
            quality: 0.0,
            qual_min: 50.0,
            ao: 0.0,
            ao_divergence_enabled: true,
            mean_reversion: false,
            tp_pct: 0.004,
            fee_taker: 0.0006,
            fee_slip: 0.0001,
            cnn_enabled: false,
            cnn_result: None,
            min_confidence: 0.60,
            open_assets: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn side_round_trips() {
        assert_eq!(Side::parse("BUY"), Some(Side::Buy));
        assert_eq!(Side::parse(" long "), Some(Side::Buy));
        assert_eq!(Side::parse("Sell"), Some(Side::Sell));
        assert_eq!(Side::parse("short"), Some(Side::Sell));
        assert_eq!(Side::parse("flat"), None);
        assert_eq!(Side::Buy.as_str(), "buy");
        assert_eq!(Side::Sell.as_str(), "sell");
    }

    #[test]
    fn cnn_vote_agreement() {
        let v = CnnVote::new(Side::Buy, 0.8);
        assert!(v.agrees_with(Side::Buy));
        assert!(!v.agrees_with(Side::Sell));
    }

    #[test]
    fn context_defaults_match_python_fallbacks() {
        let c = GateContext::default();
        assert!(c.can_trade);
        assert_eq!(c.vol_entry_min, 0.15);
        assert_eq!(c.vol_entry_max, 0.90);
        assert_eq!(c.qual_min, 50.0);
        assert_eq!(c.tp_pct, 0.004);
        assert_eq!(c.fee_taker, 0.0006);
        assert_eq!(c.fee_slip, 0.0001);
        assert_eq!(c.min_confidence, 0.60);
    }
}
