//! Build script for JANUS forward service.
//!
//! All proto compilation has been migrated out of this service:
//! - `JanusService` (signal generation, package `janus.v1`) was confirmed
//!   dead code by the 2026-03-26 STRUCT-C audit and was removed in the
//!   fks-proto decoupling pass.
//! - `regime_bridge` types live at `fks_proto::janus::regime_bridge::*`
//!   (compiled by the proto crate's own build.rs).
//! - Execution + common types live at `fks_proto::execution::*` and
//!   `fks_proto::common::*`.
//!
//! Until any local protos are added back, this script is a no-op.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    Ok(())
}
