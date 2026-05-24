//! Janus-local reconstruction of the FKS protobuf surface.
//!
//! Modules are arranged to match the Rust import paths the rest of the
//! workspace already uses (`fks_proto::common::*`, `fks_proto::execution::*`,
//! `fks_proto::feedback::*`, `fks_proto::janus::regime_bridge::*`,
//! `fks_proto::neuromorphic::distributed::*`).

#![allow(clippy::all)]
#![allow(non_camel_case_types)]
#![allow(rustdoc::broken_intra_doc_links)]

pub mod common {
    tonic::include_proto!("fks.common.v1");
}

pub mod execution {
    tonic::include_proto!("fks.execution.v1");
}

pub mod feedback {
    tonic::include_proto!("fks.feedback.v1");
}

pub mod janus {
    pub mod regime_bridge {
        tonic::include_proto!("fks.janus.v1.bridge");
    }
}

pub mod neuromorphic {
    pub mod distributed {
        tonic::include_proto!("fks.neuromorphic.distributed.v1");
    }
}

// ── Builder helpers on RegimeState ────────────────────────────────────────
//
// `regime_bridge_proto::make_push_request` in the forward service relies on
// these inherent methods. They lived in the upstream fks-proto crate; keeping
// them here preserves the call sites without changes.

impl janus::regime_bridge::RegimeState {
    /// Set the sequence counter.
    pub fn with_sequence(mut self, sequence: i64) -> Self {
        self.sequence = sequence;
        self
    }

    /// Set the wall-clock timestamp (microseconds).
    pub fn with_timestamp_us(mut self, timestamp_us: i64) -> Self {
        self.timestamp_us = timestamp_us;
        self
    }

    /// Flag this state as a regime transition and record what the regime
    /// was beforehand.
    pub fn with_transition(
        mut self,
        previous_hypothalamus: janus::regime_bridge::HypothalamusRegime,
        previous_amygdala: janus::regime_bridge::AmygdalaRegime,
    ) -> Self {
        self.is_transition = true;
        self.previous_hypothalamus_regime = previous_hypothalamus as i32;
        self.previous_amygdala_regime = previous_amygdala as i32;
        self
    }
}
