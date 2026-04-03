//! Build script for FKS Execution Service
//!
//! Proto compilation is now handled by the centralized fks-proto crate.
//! This build script is kept for any future non-proto build requirements.

fn main() {
    // Proto compilation is now handled by fks-proto crate
    // The execution service imports proto types from fks_proto::execution
    //
    // If you need to compile local protos, uncomment:
    // tonic_prost_build::compile_protos("proto/fks/execution/v1/execution.proto")?;
    // println!("cargo:rerun-if-changed=proto/fks/execution/v1/execution.proto");

    println!("cargo:info=Using centralized fks-proto crate for proto definitions");
}
