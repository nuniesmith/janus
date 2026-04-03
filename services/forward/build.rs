//! Build script for JANUS forward service
//!
//! Compiles protobuf definitions for the local janus.v1 service.
//! Execution proto types are now imported from the centralized fks-proto crate.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // JanusService (signal generation) — local compile DEFERRED (STRUCT-C).
    //
    // Audited 2026-03-26: GrpcServer (api/grpc.rs) is compiled in but never
    // started in main.rs — no GrpcServer::new() / ::start() call exists
    // anywhere outside of api/grpc.rs itself.  Consolidation is safe to do
    // but must wait until the service ownership question is resolved:
    //
    //   Option A — keep in this binary: wire GrpcServer::start() into main.rs,
    //              then move proto to proto/fks/janus/v1/signal_service.proto
    //              (package fks.janus.v1) and update tonic::include_proto! in
    //              api/grpc.rs.
    //   Option B — extract to a dedicated janus-signal service and remove
    //              api/grpc.rs from this crate entirely.
    //
    // Until that decision is made, keep compiling locally so the impl stays
    // type-checked and doesn't rot.
    tonic_prost_build::compile_protos("proto/janus/v1/janus.proto")?;
    println!("cargo:rerun-if-changed=proto/janus/v1/janus.proto");

    // regime_bridge.proto has been centralized into fks-proto
    // Types are now at fks_proto::janus::regime_bridge::*
    // See: proto/fks/janus/v1/regime_bridge.proto (package fks.janus.v1.bridge)
    // No local compile needed — fks-proto build.rs handles it.

    // Execution service proto is now provided by fks-proto crate
    // Import execution types using: use fks_proto::execution::*;

    Ok(())
}
