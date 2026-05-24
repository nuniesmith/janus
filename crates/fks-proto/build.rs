//! Compile the bundled .proto files into Rust types via tonic.

use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let proto_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("proto");

    let files = [
        "fks/common/v1/common.proto",
        "fks/execution/v1/execution.proto",
        "fks/feedback/v1/feedback.proto",
        "fks/janus/v1/regime_bridge.proto",
        "fks/neuromorphic/distributed/v1/distributed.proto",
    ];

    let inputs: Vec<PathBuf> = files.iter().map(|f| proto_root.join(f)).collect();

    tonic_prost_build::configure()
        .build_client(true)
        .build_server(true)
        .compile_protos(&inputs, &[proto_root.clone()])?;

    for f in &files {
        println!("cargo:rerun-if-changed=proto/{f}");
    }

    Ok(())
}
