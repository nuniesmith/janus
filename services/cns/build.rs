//! Build script for JANUS CNS Service
//!
//! Proto compilation has been migrated to the centralized fks-proto crate.
//! The CNS service currently uses HTTP/REST endpoints via axum, not gRPC.
//! This build script is now a no-op but kept for future build-time tasks if needed.

fn main() {
    // Proto compilation moved to fks-proto crate
    // If gRPC is needed in the future, import types using: use fks_proto::cns::*;
    println!("cargo:rerun-if-changed=build.rs");
}
