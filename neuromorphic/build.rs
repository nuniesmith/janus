//! Build script for JANUS neuromorphic module
//!
//! Proto compilation has been migrated to the centralized fks-proto crate.
//! This build script is now a no-op but kept for future build-time tasks if needed.

fn main() {
    // Proto compilation moved to fks-proto crate
    // Import distributed training types using: use fks_proto::neuromorphic::distributed::*;
    println!("cargo:rerun-if-changed=build.rs");
}
