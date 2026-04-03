#![allow(clippy::doc_lazy_continuation)]
//! Generated protobuf types for the `RegimeBridgeService`.
//!
//! **STRUCT-C COMPLETE:** This module now re-exports types from the centralized
//! `fks-proto` crate rather than compiling the local proto file.
//!
//! The canonical proto definition lives at:
//!   `proto/fks/janus/v1/regime_bridge.proto`  (package `fks.janus.v1.bridge`)
//!
//! It is compiled by `src/proto/build.rs` and exposed as
//! `fks_proto::janus::regime_bridge::*`.
//!
//! All `From` impl blocks and helper methods below remain unchanged — the
//! type names (`RegimeState`, `HypothalamusRegime`, etc.) are identical
//! between the old local proto and the centralized one; only the package
//! namespace changed from `janus.v1.bridge` → `fks.janus.v1.bridge`.
//!
//! # Usage
//!
//! ```rust,ignore
//! use janus_forward::regime_bridge_proto::regime_bridge_service_client::RegimeBridgeServiceClient;
//! use janus_forward::regime_bridge_proto::{PushRegimeStateRequest, RegimeState};
//! ```

// Re-export all types from the centralized fks-proto crate.
// Previously this was: tonic::include_proto!("janus.v1.bridge");
pub use fks_proto::janus::regime_bridge::*;

use crate::regime_bridge::{
    AmygdalaRegime as RustAmygdala, BridgedRegimeState, HypothalamusRegime as RustHypothalamus,
    RegimeIndicators as RustIndicators,
};

// ============================================================================
// Hypothalamus enum conversions
// ============================================================================

impl From<&RustHypothalamus> for HypothalamusRegime {
    fn from(r: &RustHypothalamus) -> Self {
        match r {
            RustHypothalamus::StrongBullish => HypothalamusRegime::StrongBullish,
            RustHypothalamus::Bullish => HypothalamusRegime::Bullish,
            RustHypothalamus::Neutral => HypothalamusRegime::Neutral,
            RustHypothalamus::Bearish => HypothalamusRegime::Bearish,
            RustHypothalamus::StrongBearish => HypothalamusRegime::StrongBearish,
            RustHypothalamus::HighVolatility => HypothalamusRegime::HighVolatility,
            RustHypothalamus::LowVolatility => HypothalamusRegime::LowVolatility,
            RustHypothalamus::Transitional => HypothalamusRegime::Transitional,
            RustHypothalamus::Crisis => HypothalamusRegime::Crisis,
            RustHypothalamus::Unknown => HypothalamusRegime::Unknown,
        }
    }
}

impl From<RustHypothalamus> for HypothalamusRegime {
    fn from(r: RustHypothalamus) -> Self {
        (&r).into()
    }
}

impl From<HypothalamusRegime> for RustHypothalamus {
    fn from(r: HypothalamusRegime) -> Self {
        match r {
            HypothalamusRegime::StrongBullish => RustHypothalamus::StrongBullish,
            HypothalamusRegime::Bullish => RustHypothalamus::Bullish,
            HypothalamusRegime::Neutral => RustHypothalamus::Neutral,
            HypothalamusRegime::Bearish => RustHypothalamus::Bearish,
            HypothalamusRegime::StrongBearish => RustHypothalamus::StrongBearish,
            HypothalamusRegime::HighVolatility => RustHypothalamus::HighVolatility,
            HypothalamusRegime::LowVolatility => RustHypothalamus::LowVolatility,
            HypothalamusRegime::Transitional => RustHypothalamus::Transitional,
            HypothalamusRegime::Crisis => RustHypothalamus::Crisis,
            HypothalamusRegime::Unknown | HypothalamusRegime::Unspecified => {
                RustHypothalamus::Unknown
            }
        }
    }
}

// ============================================================================
// Amygdala enum conversions
// ============================================================================

impl From<&RustAmygdala> for AmygdalaRegime {
    fn from(r: &RustAmygdala) -> Self {
        match r {
            RustAmygdala::LowVolTrending => AmygdalaRegime::LowVolTrending,
            RustAmygdala::LowVolMeanReverting => AmygdalaRegime::LowVolMeanReverting,
            RustAmygdala::HighVolTrending => AmygdalaRegime::HighVolTrending,
            RustAmygdala::HighVolMeanReverting => AmygdalaRegime::HighVolMeanReverting,
            RustAmygdala::Crisis => AmygdalaRegime::Crisis,
            RustAmygdala::Transitional => AmygdalaRegime::Transitional,
            RustAmygdala::Unknown => AmygdalaRegime::Unknown,
        }
    }
}

impl From<RustAmygdala> for AmygdalaRegime {
    fn from(r: RustAmygdala) -> Self {
        (&r).into()
    }
}

impl From<AmygdalaRegime> for RustAmygdala {
    fn from(r: AmygdalaRegime) -> Self {
        match r {
            AmygdalaRegime::LowVolTrending => RustAmygdala::LowVolTrending,
            AmygdalaRegime::LowVolMeanReverting => RustAmygdala::LowVolMeanReverting,
            AmygdalaRegime::HighVolTrending => RustAmygdala::HighVolTrending,
            AmygdalaRegime::HighVolMeanReverting => RustAmygdala::HighVolMeanReverting,
            AmygdalaRegime::Crisis => RustAmygdala::Crisis,
            AmygdalaRegime::Transitional => RustAmygdala::Transitional,
            AmygdalaRegime::Unknown | AmygdalaRegime::Unspecified => RustAmygdala::Unknown,
        }
    }
}

// ============================================================================
// Indicator conversions
// ============================================================================

impl From<&RustIndicators> for RegimeIndicators {
    fn from(i: &RustIndicators) -> Self {
        RegimeIndicators {
            trend: i.trend,
            trend_strength: i.trend_strength,
            volatility: i.volatility,
            volatility_percentile: i.volatility_percentile,
            correlation: i.correlation,
            breadth: i.breadth,
            momentum: i.momentum,
            relative_volume: i.relative_volume,
            liquidity_score: i.liquidity_score,
            fear_index: i.fear_index.unwrap_or(0.0),
        }
    }
}

impl From<RegimeIndicators> for RustIndicators {
    fn from(i: RegimeIndicators) -> Self {
        RustIndicators {
            trend: i.trend,
            trend_strength: i.trend_strength,
            volatility: i.volatility,
            volatility_percentile: i.volatility_percentile,
            correlation: i.correlation,
            breadth: i.breadth,
            momentum: i.momentum,
            relative_volume: i.relative_volume,
            liquidity_score: i.liquidity_score,
            fear_index: Some(i.fear_index),
        }
    }
}

// ============================================================================
// BridgedRegimeState → RegimeState (proto) conversion
// ============================================================================

impl From<&BridgedRegimeState> for RegimeState {
    fn from(b: &BridgedRegimeState) -> Self {
        RegimeState {
            symbol: b.symbol.clone(),
            hypothalamus_regime: HypothalamusRegime::from(&b.hypothalamus_regime).into(),
            amygdala_regime: AmygdalaRegime::from(&b.amygdala_regime).into(),
            position_scale: b.position_scale,
            is_high_risk: b.is_high_risk,
            confidence: b.confidence,
            indicators: Some(RegimeIndicators::from(&b.indicators)),
            timestamp_us: chrono::Utc::now().timestamp_micros(),
            sequence: 0,
            is_transition: false,
            previous_hypothalamus_regime: HypothalamusRegime::Unspecified.into(),
            previous_amygdala_regime: AmygdalaRegime::Unspecified.into(),
        }
    }
}

impl From<BridgedRegimeState> for RegimeState {
    fn from(b: BridgedRegimeState) -> Self {
        (&b).into()
    }
}

// ============================================================================
// Free-function constructors for push requests
//
// These cannot be inherent methods on `PushRegimeStateRequest` /
// `PushRegimeStateBatchRequest` because those types are defined in `fks-proto`
// (a separate crate).  Rust's orphan rule (E0116) forbids adding inherent
// impls to types from another crate.
//
// The builder methods that operate purely on `RegimeState` fields
// (`with_sequence`, `with_transition`, `with_timestamp_us`) live in
// `fks-proto/src/lib.rs` where the type is defined.
// ============================================================================

/// Build a [`PushRegimeStateRequest`] from a [`BridgedRegimeState`].
pub fn make_push_request(
    state: &BridgedRegimeState,
    source_id: impl Into<String>,
) -> PushRegimeStateRequest {
    PushRegimeStateRequest {
        state: Some(RegimeState::from(state)),
        source_id: source_id.into(),
    }
}

/// Build a [`PushRegimeStateBatchRequest`] from a slice of [`BridgedRegimeState`].
pub fn make_push_batch_request(
    states: &[BridgedRegimeState],
    source_id: impl Into<String>,
) -> PushRegimeStateBatchRequest {
    PushRegimeStateBatchRequest {
        states: states.iter().map(RegimeState::from).collect(),
        source_id: source_id.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regime_bridge::{
        AmygdalaRegime as RUSTCODE, BridgedRegimeState, HypothalamusRegime as RH, RegimeIndicators as RI,
    };

    fn sample_bridged_state() -> BridgedRegimeState {
        BridgedRegimeState {
            symbol: "BTCUSD".to_string(),
            hypothalamus_regime: RH::StrongBullish,
            amygdala_regime: RUSTCODE::LowVolTrending,
            position_scale: 1.2,
            is_high_risk: false,
            confidence: 0.85,
            indicators: RI {
                trend: 0.75,
                trend_strength: 0.6,
                volatility: 500.0,
                volatility_percentile: 0.3,
                correlation: 0.5,
                breadth: 0.5,
                momentum: 0.4,
                relative_volume: 1.5,
                liquidity_score: 0.9,
                fear_index: Some(0.1),
            },
        }
    }

    #[test]
    fn test_hypothalamus_roundtrip() {
        let variants = vec![
            RH::StrongBullish,
            RH::Bullish,
            RH::Neutral,
            RH::Bearish,
            RH::StrongBearish,
            RH::HighVolatility,
            RH::LowVolatility,
            RH::Transitional,
            RH::Crisis,
            RH::Unknown,
        ];
        for v in variants {
            let proto: HypothalamusRegime = v.into();
            let back: RH = proto.into();
            assert_eq!(
                format!("{}", v),
                format!("{}", back),
                "roundtrip failed for {:?}",
                v
            );
        }
    }

    #[test]
    fn test_amygdala_roundtrip() {
        let variants = vec![
            RUSTCODE::LowVolTrending,
            RUSTCODE::LowVolMeanReverting,
            RUSTCODE::HighVolTrending,
            RUSTCODE::HighVolMeanReverting,
            RUSTCODE::Crisis,
            RUSTCODE::Transitional,
            RUSTCODE::Unknown,
        ];
        for v in variants {
            let proto: AmygdalaRegime = v.into();
            let back: RUSTCODE = proto.into();
            assert_eq!(
                format!("{}", v),
                format!("{}", back),
                "roundtrip failed for {:?}",
                v
            );
        }
    }

    #[test]
    fn test_indicators_roundtrip() {
        let orig = RI {
            trend: 0.75,
            trend_strength: 0.6,
            volatility: 500.0,
            volatility_percentile: 0.3,
            correlation: 0.5,
            breadth: 0.5,
            momentum: 0.4,
            relative_volume: 1.5,
            liquidity_score: 0.9,
            fear_index: Some(0.1),
        };
        let proto: RegimeIndicators = (&orig).into();
        let back: RI = proto.into();
        assert!((back.trend - orig.trend).abs() < f64::EPSILON);
        assert!((back.trend_strength - orig.trend_strength).abs() < f64::EPSILON);
        assert!((back.volatility - orig.volatility).abs() < f64::EPSILON);
        assert!((back.relative_volume - orig.relative_volume).abs() < f64::EPSILON);
        assert!((back.fear_index.unwrap() - orig.fear_index.unwrap()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_bridged_state_to_regime_state() {
        let bridged = sample_bridged_state();
        let proto: RegimeState = (&bridged).into();

        assert_eq!(proto.symbol, "BTCUSD");
        assert_eq!(
            proto.hypothalamus_regime,
            i32::from(HypothalamusRegime::StrongBullish)
        );
        assert_eq!(
            proto.amygdala_regime,
            i32::from(AmygdalaRegime::LowVolTrending)
        );
        assert!((proto.position_scale - 1.2).abs() < f64::EPSILON);
        assert!(!proto.is_high_risk);
        assert!((proto.confidence - 0.85).abs() < f64::EPSILON);
        assert!(!proto.is_transition);
        assert!(proto.indicators.is_some());

        let ind = proto.indicators.unwrap();
        assert!((ind.trend - 0.75).abs() < f64::EPSILON);
        assert!((ind.relative_volume - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_push_request_from_bridged() {
        let bridged = sample_bridged_state();
        let req = super::make_push_request(&bridged, "test-consumer-1");

        assert_eq!(req.source_id, "test-consumer-1");
        assert!(req.state.is_some());
        assert_eq!(req.state.unwrap().symbol, "BTCUSD");
    }

    #[test]
    fn test_batch_request_from_bridged() {
        let states = vec![sample_bridged_state(), sample_bridged_state()];
        let req = super::make_push_batch_request(&states, "batch-src");

        assert_eq!(req.source_id, "batch-src");
        assert_eq!(req.states.len(), 2);
    }

    #[test]
    fn test_regime_state_with_transition() {
        let bridged = sample_bridged_state();
        let state = RegimeState::from(&bridged).with_transition(
            HypothalamusRegime::Neutral,
            AmygdalaRegime::LowVolMeanReverting,
        );

        assert!(state.is_transition);
        assert_eq!(
            state.previous_hypothalamus_regime,
            i32::from(HypothalamusRegime::Neutral)
        );
        assert_eq!(
            state.previous_amygdala_regime,
            i32::from(AmygdalaRegime::LowVolMeanReverting)
        );
    }

    #[test]
    fn test_regime_state_with_sequence() {
        let bridged = sample_bridged_state();
        let state = RegimeState::from(&bridged).with_sequence(42);
        assert_eq!(state.sequence, 42);
    }

    #[test]
    fn test_unspecified_maps_to_unknown() {
        let hypo: RH = HypothalamusRegime::Unspecified.into();
        assert!(matches!(hypo, RH::Unknown));

        let amyg: RUSTCODE = AmygdalaRegime::Unspecified.into();
        assert!(matches!(amyg, RUSTCODE::Unknown));
    }
}
