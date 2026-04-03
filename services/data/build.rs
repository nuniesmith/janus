//! Build script for Data Service
//!
//! Proto compilation has been migrated to the centralized fks-proto crate.
//! The Data service currently uses pre-generated proto files in src/proto/
//! and HTTP/WebSocket endpoints via axum, not gRPC.
//! This build script is now a no-op but kept for future build-time tasks if needed.

fn main() {
    // Proto compilation moved to fks-proto crate
    // If gRPC is needed in the future, import types using: use fks_proto::data::*;
    println!("cargo:rerun-if-changed=build.rs");
}
