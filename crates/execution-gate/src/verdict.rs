//! Gate verdicts — the result of running a trade-entry through the gate chain.

use serde::{Deserialize, Serialize};

/// Result of an execution-gate evaluation.
///
/// Faithful port of Ruby's `GateVerdict` enum
/// (`ruby/src/services/execution_gate.py`). The chain runs gates in priority
/// order and returns the first non-`Pass` verdict it encounters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GateVerdict {
    /// All gates cleared — the entry is allowed.
    Pass,
    /// Consecutive-loss circuit breaker is open (cooling down).
    BlockCircuitBreaker,
    /// Risk engine forbids trading (daily loss / drawdown / exposure).
    BlockRisk,
    /// Realised volatility is outside the entry band.
    BlockVolFilter,
    /// Signal-quality score below the minimum.
    BlockQuality,
    /// Awesome-Oscillator sign diverges from the signal direction.
    BlockAoDivergence,
    /// Fees/slippage too large relative to the take-profit target.
    BlockFeeViability,
    /// CNN inference disagrees with the proposed entry side.
    BlockCnnDisagree,
    /// CNN confidence below the (possibly adaptive) threshold.
    BlockCnnConfidence,
    /// Too many open positions already correlated with this asset.
    BlockCorrelation,
}

impl GateVerdict {
    /// `true` for [`GateVerdict::Pass`].
    pub fn is_pass(self) -> bool {
        matches!(self, GateVerdict::Pass)
    }

    /// `true` for any blocking verdict.
    pub fn is_block(self) -> bool {
        !self.is_pass()
    }

    /// Lowercased label used for per-reason metric keys — mirrors Python's
    /// `GateVerdict.name.lower()` so dashboards stay wire-compatible
    /// (e.g. `block_circuit_breaker`).
    pub fn as_counter_label(self) -> &'static str {
        match self {
            GateVerdict::Pass => "pass",
            GateVerdict::BlockCircuitBreaker => "block_circuit_breaker",
            GateVerdict::BlockRisk => "block_risk",
            GateVerdict::BlockVolFilter => "block_vol_filter",
            GateVerdict::BlockQuality => "block_quality",
            GateVerdict::BlockAoDivergence => "block_ao_divergence",
            GateVerdict::BlockFeeViability => "block_fee_viability",
            GateVerdict::BlockCnnDisagree => "block_cnn_disagree",
            GateVerdict::BlockCnnConfidence => "block_cnn_confidence",
            GateVerdict::BlockCorrelation => "block_correlation",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pass_is_not_a_block() {
        assert!(GateVerdict::Pass.is_pass());
        assert!(!GateVerdict::Pass.is_block());
    }

    #[test]
    fn blocks_are_blocks() {
        for v in [
            GateVerdict::BlockCircuitBreaker,
            GateVerdict::BlockRisk,
            GateVerdict::BlockVolFilter,
            GateVerdict::BlockQuality,
            GateVerdict::BlockAoDivergence,
            GateVerdict::BlockFeeViability,
            GateVerdict::BlockCnnDisagree,
            GateVerdict::BlockCnnConfidence,
            GateVerdict::BlockCorrelation,
        ] {
            assert!(v.is_block());
            assert!(!v.is_pass());
        }
    }

    #[test]
    fn counter_labels_match_python_name_lower() {
        // These strings are part of the dashboard contract — keep stable.
        assert_eq!(
            GateVerdict::BlockCircuitBreaker.as_counter_label(),
            "block_circuit_breaker"
        );
        assert_eq!(
            GateVerdict::BlockCnnDisagree.as_counter_label(),
            "block_cnn_disagree"
        );
        assert_eq!(
            GateVerdict::BlockCorrelation.as_counter_label(),
            "block_correlation"
        );
    }
}
