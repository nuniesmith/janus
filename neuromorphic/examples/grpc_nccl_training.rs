//! gRPC + NCCL Distributed Training Example
//!
//! This example demonstrates how to use the gRPC backend for inter-node communication
//! combined with NCCL for intra-node GPU communication in a multi-node, multi-GPU
//! distributed training setup.
//!
//! # Architecture
//!
//! - Parameter Server (gRPC): Coordinates gradient aggregation across nodes
//! - NCCL: Handles GPU-to-GPU communication within each node
//! - Hybrid Strategy: NCCL for local GPUs, gRPC for cross-node sync
//!
//! # Usage
//!
//! ## Start Parameter Server
//! ```bash
//! cargo run --example grpc_nccl_training -- --mode server
//! ```
//!
//! ## Start Worker Nodes
//! ```bash
//! # Node 0 (4 GPUs)
//! CUDA_VISIBLE_DEVICES=0,1,2,3 cargo run --release --features nccl \
//!     --example grpc_nccl_training -- \
//!     --mode worker \
//!     --rank 0 \
//!     --world-size 2 \
//!     --local-gpus 4 \
//!     --master-addr localhost:50051
//!
//! # Node 1 (4 GPUs)
//! CUDA_VISIBLE_DEVICES=0,1,2,3 cargo run --release --features nccl \
//!     --example grpc_nccl_training -- \
//!     --mode worker \
//!     --rank 1 \
//!     --world-size 2 \
//!     --local-gpus 4 \
//!     --master-addr localhost:50051
//! ```

use anyhow::{Result, anyhow};
use std::env;
use std::time::{Duration, Instant};
use tokio::time::sleep;
use tracing::info;
use tracing_subscriber::{EnvFilter, fmt, prelude::*};

#[cfg(feature = "nccl")]
use janus_neuromorphic::distributed::{
    DistributedTrainingClient, DistributedTrainingServer, NcclBackend, NcclConfig, NcclReduceOp,
};

#[cfg(not(feature = "nccl"))]
use janus_neuromorphic::distributed::{DistributedTrainingClient, DistributedTrainingServer};

/// CLI arguments
#[derive(Debug, Clone)]
struct Args {
    /// Mode: server or worker
    mode: Mode,
    /// Node rank (for workers)
    rank: usize,
    /// Total number of nodes
    world_size: usize,
    /// Number of local GPUs
    local_gpus: usize,
    /// Master server address
    master_addr: String,
    /// Number of training steps
    num_steps: usize,
    /// Model size (number of parameters)
    model_size: usize,
    /// Batch size per GPU
    batch_size: usize,
}

#[derive(Debug, Clone, PartialEq)]
enum Mode {
    Server,
    Worker,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            mode: Mode::Worker,
            rank: 0,
            world_size: 1,
            local_gpus: 1,
            master_addr: "localhost:50051".to_string(),
            num_steps: 100,
            model_size: 1_000_000, // 1M parameters
            batch_size: 32,
        }
    }
}

impl Args {
    fn from_env() -> Self {
        let mut args = Args::default();

        let env_args: Vec<String> = env::args().collect();
        let mut i = 1;
        while i < env_args.len() {
            match env_args[i].as_str() {
                "--mode" => {
                    i += 1;
                    args.mode = match env_args[i].as_str() {
                        "server" => Mode::Server,
                        "worker" => Mode::Worker,
                        _ => Mode::Worker,
                    };
                }
                "--rank" => {
                    i += 1;
                    args.rank = env_args[i].parse().unwrap_or(0);
                }
                "--world-size" => {
                    i += 1;
                    args.world_size = env_args[i].parse().unwrap_or(1);
                }
                "--local-gpus" => {
                    i += 1;
                    args.local_gpus = env_args[i].parse().unwrap_or(1);
                }
                "--master-addr" => {
                    i += 1;
                    args.master_addr = env_args[i].clone();
                }
                "--num-steps" => {
                    i += 1;
                    args.num_steps = env_args[i].parse().unwrap_or(100);
                }
                "--model-size" => {
                    i += 1;
                    args.model_size = env_args[i].parse().unwrap_or(1_000_000);
                }
                "--batch-size" => {
                    i += 1;
                    args.batch_size = env_args[i].parse().unwrap_or(32);
                }
                _ => {}
            }
            i += 1;
        }

        // Override with environment variables if present
        if let Ok(rank) = env::var("RANK") {
            args.rank = rank.parse().unwrap_or(0);
        }
        if let Ok(world_size) = env::var("WORLD_SIZE") {
            args.world_size = world_size.parse().unwrap_or(1);
        }
        if let Ok(master) = env::var("MASTER_ADDR") {
            args.master_addr = master;
        }

        args
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(EnvFilter::from_default_env().add_directive(tracing::Level::INFO.into()))
        .init();

    let args = Args::from_env();

    info!("=== gRPC + NCCL Distributed Training ===");
    info!("Mode: {:?}", args.mode);

    match args.mode {
        Mode::Server => run_server(&args).await,
        Mode::Worker => run_worker(&args).await,
    }
}

/// Run parameter server
async fn run_server(args: &Args) -> Result<()> {
    info!("Starting gRPC parameter server on {}", args.master_addr);

    let server = DistributedTrainingServer::new(&args.master_addr);
    server.serve().await?;

    Ok(())
}

/// Run worker node with NCCL + gRPC
async fn run_worker(args: &Args) -> Result<()> {
    info!(
        "Starting worker: rank={}, world_size={}, local_gpus={}",
        args.rank, args.world_size, args.local_gpus
    );

    // Connect to parameter server
    let master_url = if args.master_addr.starts_with("http") {
        args.master_addr.clone()
    } else {
        format!("http://{}", args.master_addr)
    };

    info!("Connecting to parameter server: {}", master_url);
    let mut grpc_client = DistributedTrainingClient::connect(&master_url).await?;
    grpc_client.set_rank(args.rank as u32);

    info!("Connected to parameter server");

    // Register node
    #[cfg(feature = "nccl")]
    let capabilities = janus_neuromorphic::distributed::grpc_client::create_node_capabilities(
        true,
        true,
        args.local_gpus as u32,
        "NVIDIA GPU".to_string(),
    );

    #[cfg(not(feature = "nccl"))]
    let capabilities = janus_neuromorphic::distributed::grpc_client::create_node_capabilities(
        false,
        false,
        0,
        "CPU".to_string(),
    );

    let hostname = hostname::get()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();

    let registration = grpc_client
        .register_node(
            args.rank as u32,
            hostname.clone(),
            "127.0.0.1".to_string(),
            (0..args.local_gpus as u32).collect(),
            capabilities,
        )
        .await?;

    info!(
        "Node registered: assigned_rank={}, world_size={}",
        registration.assigned_rank, registration.world_size
    );

    // Initialize NCCL backend for local GPUs
    #[cfg(feature = "nccl")]
    let _nccl_backends = initialize_nccl_backends(args).await?;

    #[cfg(not(feature = "nccl"))]
    let _nccl_backends = Vec::<()>::new();

    info!("NCCL backends initialized for {} GPUs", args.local_gpus);

    // Wait at barrier for all nodes to be ready
    info!("Waiting at initialization barrier...");
    grpc_client.barrier(0, args.rank as u32).await?;
    info!("All nodes ready, starting training");

    // Training loop
    let total_params = args.model_size;
    let mut total_step_time = Duration::ZERO;
    let mut total_sync_time = Duration::ZERO;

    for step in 0..args.num_steps {
        let step_start = Instant::now();

        // Simulate forward + backward pass (generate dummy gradients)
        let local_gradients = generate_dummy_gradients(total_params, args.rank, step);

        // Step 1: Sync gradients across local GPUs using NCCL
        #[cfg(feature = "nccl")]
        let local_synced = sync_local_nccl(&nccl_backends, local_gradients).await?;

        #[cfg(not(feature = "nccl"))]
        let local_synced = local_gradients;

        // Step 2: Sync gradients across nodes using gRPC
        let sync_start = Instant::now();
        let global_synced = sync_global_grpc(&mut grpc_client, args.rank, local_synced).await?;
        let sync_time = sync_start.elapsed();
        total_sync_time += sync_time;

        let step_time = step_start.elapsed();
        total_step_time += step_time;

        // Log progress
        if (step + 1) % 10 == 0 {
            let avg_step_time = total_step_time.as_secs_f64() / (step + 1) as f64;
            let avg_sync_time = total_sync_time.as_secs_f64() / (step + 1) as f64;
            let throughput = args.batch_size as f64 * args.local_gpus as f64 / avg_step_time;

            info!(
                "Step {}/{}: step_time={:.2}ms, sync_time={:.2}ms, throughput={:.0} samples/s",
                step + 1,
                args.num_steps,
                avg_step_time * 1000.0,
                avg_sync_time * 1000.0,
                throughput
            );
        }

        // Simulate parameter update
        let _updated_params = apply_gradients(&global_synced);

        // Send heartbeat every 10 steps
        if (step + 1) % 10 == 0 {
            let resources = janus_neuromorphic::distributed::grpc_client::create_resource_usage();
            let _ = grpc_client
                .heartbeat(args.rank as u32, 3, Some(resources))
                .await;
        }
    }

    // Final barrier
    info!("Training complete, waiting at final barrier...");
    grpc_client.barrier(1, args.rank as u32).await?;

    // Print summary
    let avg_step_time = total_step_time.as_secs_f64() / args.num_steps as f64;
    let avg_sync_time = total_sync_time.as_secs_f64() / args.num_steps as f64;
    let total_throughput = args.batch_size as f64 * args.local_gpus as f64 / avg_step_time;

    info!("=== Training Summary ===");
    info!("Total steps: {}", args.num_steps);
    info!("Avg step time: {:.2}ms", avg_step_time * 1000.0);
    info!("Avg sync time: {:.2}ms", avg_sync_time * 1000.0);
    info!("Throughput: {:.0} samples/s", total_throughput);
    info!(
        "Communication overhead: {:.1}%",
        (avg_sync_time / avg_step_time) * 100.0
    );

    Ok(())
}

/// Initialize NCCL backends for all local GPUs
#[cfg(feature = "nccl")]
async fn initialize_nccl_backends(args: &Args) -> Result<Vec<Arc<NcclBackend>>> {
    use janus_neuromorphic::distributed::NcclBackend;

    info!("Initializing NCCL for {} GPUs", args.local_gpus);

    let mut backends = Vec::new();

    // Generate NCCL unique ID on rank 0, broadcast to others
    // For simplicity, we'll create independent communicators per GPU
    // In production, you'd coordinate the unique ID across all nodes

    for gpu_id in 0..args.local_gpus {
        let config = NcclConfig {
            rank: gpu_id,
            world_size: args.local_gpus,
            device_id: gpu_id,
            nccl_id: None,
            enable_profiling: false,
            network_interface: None,
            use_rdma: false,
        };

        match NcclBackend::new(config).await {
            Ok(backend) => {
                info!("NCCL backend initialized for GPU {}", gpu_id);
                backends.push(Arc::new(backend));
            }
            Err(e) => {
                warn!("Failed to initialize NCCL for GPU {}: {}", gpu_id, e);
                warn!("Falling back to CPU-only gradient sync");
            }
        }
    }

    if backends.is_empty() {
        return Err(anyhow!("Failed to initialize any NCCL backends"));
    }

    Ok(backends)
}

/// Sync gradients across local GPUs using NCCL
#[cfg(feature = "nccl")]
async fn sync_local_nccl(backends: &[Arc<NcclBackend>], gradients: Vec<f32>) -> Result<Vec<f32>> {
    if backends.is_empty() {
        return Ok(gradients);
    }

    // Use first backend to perform all-reduce
    let backend = &backends[0];
    let synced = backend.all_reduce(gradients, NcclReduceOp::Avg).await?;

    Ok(synced)
}

/// Sync gradients across nodes using gRPC
async fn sync_global_grpc(
    client: &mut DistributedTrainingClient,
    rank: usize,
    gradients: Vec<f32>,
) -> Result<Vec<f32>> {
    // Push gradients to parameter server
    let push_response = client
        .push_gradients("model.gradients", gradients, 0, rank as u32)
        .await?;

    if !push_response.success {
        return Err(anyhow!(
            "Failed to push gradients: {}",
            push_response.message
        ));
    }

    // Small delay to allow other nodes to push
    sleep(Duration::from_millis(10)).await;

    // Pull updated parameters
    let pull_response = client
        .pull_parameters("model.gradients", rank as u32)
        .await?;

    if !pull_response.success {
        return Err(anyhow!(
            "Failed to pull parameters: {}",
            pull_response.message
        ));
    }

    Ok(pull_response.data)
}

/// Generate dummy gradients for testing
fn generate_dummy_gradients(size: usize, rank: usize, step: usize) -> Vec<f32> {
    let mut gradients = Vec::with_capacity(size);
    let base = (rank * 1000 + step) as f32;

    for i in 0..size {
        // Create some variation in gradients
        let val = (base + i as f32) * 0.001;
        gradients.push(val.sin());
    }

    gradients
}

/// Apply gradients to parameters (dummy implementation)
fn apply_gradients(gradients: &[f32]) -> Vec<f32> {
    // In real training, this would update model parameters
    // For now, just return a dummy result
    gradients.iter().map(|g| g * 0.9).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_args_parsing() {
        let args = Args::default();
        assert_eq!(args.rank, 0);
        assert_eq!(args.world_size, 1);
    }

    #[test]
    fn test_gradient_generation() {
        let grads = generate_dummy_gradients(100, 0, 0);
        assert_eq!(grads.len(), 100);
    }

    #[test]
    fn test_apply_gradients() {
        let grads = vec![1.0, 2.0, 3.0];
        let updated = apply_gradients(&grads);
        assert_eq!(updated.len(), 3);
        assert!((updated[0] - 0.9).abs() < 1e-6);
    }
}
