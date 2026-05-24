//! Janus-local reconstruction of the FKS protobuf surface.
//!
//! The proto packages are deeply qualified (`fks.execution.v1`,
//! `fks.common.v1`, …) so that cross-package field references like
//! `fks.common.v1.Side` resolve correctly in the prost-generated code.
//! Consumers don't need to know about that — the flat re-exports at the
//! bottom of this file (`fks_proto::common`, `fks_proto::execution`, …)
//! preserve the import surface the rest of the workspace already uses.

#![allow(clippy::all)]
#![allow(non_camel_case_types)]
#![allow(rustdoc::broken_intra_doc_links)]

pub mod fks {
    pub mod common {
        pub mod v1 {
            tonic::include_proto!("fks.common.v1");
        }
    }
    pub mod execution {
        pub mod v1 {
            tonic::include_proto!("fks.execution.v1");
        }
    }
    pub mod feedback {
        pub mod v1 {
            tonic::include_proto!("fks.feedback.v1");
        }
    }
    pub mod janus {
        pub mod v1 {
            pub mod bridge {
                tonic::include_proto!("fks.janus.v1.bridge");
            }
        }
    }
    pub mod neuromorphic {
        pub mod distributed {
            pub mod v1 {
                tonic::include_proto!("fks.neuromorphic.distributed.v1");
            }
        }
    }
}

// ── Flat re-exports matching the existing call sites ──────────────────────
//
// Everywhere in the workspace already uses `fks_proto::common::*`,
// `fks_proto::execution::*`, `fks_proto::feedback::*`,
// `fks_proto::janus::regime_bridge::*`, and
// `fks_proto::neuromorphic::distributed::*`. These re-exports keep those
// paths working without churning every import.

pub use fks::common::v1 as common;
pub use fks::execution::v1 as execution;
pub use fks::feedback::v1 as feedback;

pub mod janus {
    pub use crate::fks::janus::v1::bridge as regime_bridge;
}

pub mod neuromorphic {
    pub use crate::fks::neuromorphic::distributed::v1 as distributed;
}

// ── Builder helpers on RegimeState ────────────────────────────────────────
//
// `regime_bridge_proto::make_push_request` in the forward service relies
// on these inherent methods. They lived in the upstream fks-proto crate;
// keeping them here preserves the call sites unchanged.

impl fks::janus::v1::bridge::RegimeState {
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
        previous_hypothalamus: fks::janus::v1::bridge::HypothalamusRegime,
        previous_amygdala: fks::janus::v1::bridge::AmygdalaRegime,
    ) -> Self {
        self.is_transition = true;
        self.previous_hypothalamus_regime = previous_hypothalamus as i32;
        self.previous_amygdala_regime = previous_amygdala as i32;
        self
    }
}
