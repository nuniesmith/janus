//! Multi-Node Distributed Training Example
//!
//! This example demonstrates how to launch and coordinate training across multiple nodes.
//! It shows the complete workflow from node initialization to distributed gradient
//! synchronization and coordinated checkpointing.
//!
//! # Usage
//!
//! Launch on each node with appropriate rank:
//!
//! ```bash
//! # On node 0 (master)
//! cargo run --example train_multinode -- \
//!     --rank 0 \
//!     --world-size 4 \
//!     --master-addr "192.168.1.100:50051" \
//!     --node-addrs "192.168.1.100:50051,192.168.1.101:50051,192.168.1.102:50051,192.168.1.103:50051"
//!
//! # On node 1 (worker)
//! cargo run --example train_multinode -- \
//!     --rank 1 \
//!     --world-size 4 \
//!     --master-addr "192.168.1.100:50051" \
//!     --node-addrs "192.168.1.100:50051,192.168.1.101:50051,192.168.1.102:50051,192.168.1.103:50051"
//!
//! # Repeat for nodes 2 and 3...
//! ```
//!
//! # Environment Variables
//!
//! Alternatively, use environment variables:
//! - `RANK`: Node rank (0 = master)
//! - `WORLD_SIZE`: Total number of nodes
//! - `MASTER_ADDR`: Master node address
//! - `NODE_ADDRS`: Comma-separated list of all node addresses
//!
//! ```bash
//! export RANK=0
//! export WORLD_SIZE=4
//! export MASTER_ADDR="192.168.1.100:50051"
//! export NODE_ADDRS="192.168.1.100:50051,192.168.1.101:50051,192.168.1.102:50051,192.168.1.103:50051"
//! cargo run --example train_multinode
//! ```

use anyhow::Result;
use janus_neuromorphic::distributed::{
    MultiNodeCheckpoint, MultiNodeConfig, MultiNodeCoordinator, SyncMethod,
};
use janus_neuromorphic::integration::{Candle, GafFeature, MarketDataPipeline, PipelineConfig};
use std::env;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tracing::{info, warn};
use tracing_subscriber::{EnvFilter, fmt, prelude::*};

/// CLI arguments for multi-node training
#[derive(Debug, Clone)]
struct Args {
    /// Node rank (0 = master)
    rank: usize,
    /// Total number of nodes
    world_size: usize,
    /// Master node address
    master_addr: String,
    /// All node addresses (comma-separated)
    node_addrs: Vec<String>,
    /// Number of epochs to train
    epochs: usize,
    /// Batch size per device
    batch_size: usize,
    /// Learning rate
    learning_rate: f64,
    /// Sync method
    sync_method: SyncMethod,
    /// Path to market data
    data_path: Option<PathBuf>,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            rank: 0,
            world_size: 1,
            master_addr: "localhost:50051".to_string(),
            node_addrs: vec!["localhost:50051".to_string()],
            epochs: 10,
            batch_size: 16,
            learning_rate: 1e-4,
            sync_method: SyncMethod::AllReduce,
            data_path: None,
        }
    }
}

impl Args {
    /// Parse from environment variables
    fn from_env() -> Result<Self> {
        let mut args = Args::default();

        if let Ok(rank) = env::var("RANK") {
            args.rank = rank.parse()?;
        }

        if let Ok(world_size) = env::var("WORLD_SIZE") {
            args.world_size = world_size.parse()?;
        }

        if let Ok(master_addr) = env::var("MASTER_ADDR") {
            args.master_addr = master_addr;
        }

        if let Ok(node_addrs) = env::var("NODE_ADDRS") {
            args.node_addrs = node_addrs.split(',').map(|s| s.to_string()).collect();
        }

        if let Ok(epochs) = env::var("EPOCHS") {
            args.epochs = epochs.parse()?;
        }

        if let Ok(batch_size) = env::var("BATCH_SIZE") {
            args.batch_size = batch_size.parse()?;
        }

        if let Ok(lr) = env::var("LEARNING_RATE") {
            args.learning_rate = lr.parse()?;
        }

        if let Ok(data_path) = env::var("DATA_PATH") {
            args.data_path = Some(PathBuf::from(data_path));
        }

        Ok(args)
    }

    /// Validate arguments
    fn validate(&self) -> Result<()> {
        if self.rank >= self.world_size {
            return Err(anyhow::anyhow!(
                "Invalid rank: {} >= world_size: {}",
                self.rank,
                self.world_size
            ));
        }

        if self.node_addrs.len() != self.world_size {
            return Err(anyhow::anyhow!(
                "Node addresses mismatch: {} addrs for world_size {}",
                self.node_addrs.len(),
                self.world_size
            ));
        }

        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(EnvFilter::from_default_env().add_directive(tracing::Level::INFO.into()))
        .init();

    info!("=== Multi-Node Distributed Training ===");

    // Parse arguments
    let args = Args::from_env()?;
    args.validate()?;

    info!("Configuration:");
    info!("  Rank: {}/{}", args.rank, args.world_size);
    info!("  Master: {}", args.master_addr);
    info!("  Nodes: {:?}", args.node_addrs);
    info!("  Sync method: {:?}", args.sync_method);

    // Step 1: Initialize multi-node coordinator
    info!("\n=== Step 1: Initializing Multi-Node Coordinator ===");
    let coordinator = initialize_coordinator(&args).await?;

    // Step 2: Load and prepare data (on master only, then distribute)
    info!("\n=== Step 2: Loading Data ===");
    let data_pipeline = if coordinator.is_master() {
        Some(load_data(&args).await?)
    } else {
        None
    };

    // Step 3: Wait for all nodes to be ready
    info!("\n=== Step 3: Synchronizing Nodes ===");
    coordinator.barrier().await?;
    info!("All nodes ready!");

    // Step 4: Train the model
    info!("\n=== Step 4: Distributed Training ===");
    train_distributed(&coordinator, data_pipeline, &args).await?;

    // Step 5: Graceful shutdown
    info!("\n=== Step 5: Shutdown ===");
    coordinator.shutdown().await?;

    info!("\n=== Training Complete ===");
    Ok(())
}

/// Initialize multi-node coordinator
async fn initialize_coordinator(args: &Args) -> Result<Arc<MultiNodeCoordinator>> {
    let config = MultiNodeConfig {
        rank: args.rank,
        world_size: args.world_size,
        master_addr: args.master_addr.clone(),
        node_addrs: args.node_addrs.clone(),
        sync_method: args.sync_method,
        fault_tolerance: true,
        compression: true,
        sync_frequency: 1,
        timeout_secs: 300,
    };

    let coordinator = MultiNodeCoordinator::new(config).await?;

    info!("Multi-node coordinator initialized");
    info!(
        "  Local GPUs: {}",
        coordinator.local_coordinator().available_devices().len()
    );
    info!("  Total nodes: {}", coordinator.world_size());

    Ok(Arc::new(coordinator))
}

/// Load market data (master only)
async fn load_data(args: &Args) -> Result<MarketDataPipeline> {
    info!("Loading market data...");

    let config = PipelineConfig {
        num_frames: 16,
        candles_per_frame: 60,
        gaf_image_size: 224,
        features: vec![GafFeature::Close, GafFeature::Volume, GafFeature::HighLow],
        prediction_horizon: 10,
        buy_threshold: 0.5,
        sell_threshold: -0.5,
        train_split: 0.8,
        shuffle: true,
        seed: Some(42),
    };

    let mut pipeline = MarketDataPipeline::new(config);

    if let Some(data_path) = &args.data_path {
        if data_path.exists() {
            info!("Loading from: {:?}", data_path);
            pipeline.load_from_csv(data_path).await?;
        } else {
            warn!("Data file not found, using synthetic data");
            let synthetic = generate_synthetic_data(10000);
            pipeline.load_from_vec(synthetic);
        }
    } else {
        info!("No data path specified, using synthetic data");
        let synthetic = generate_synthetic_data(10000);
        pipeline.load_from_vec(synthetic);
    }

    pipeline.preprocess().await?;

    let (sell, hold, buy) = pipeline.label_distribution();
    info!("Data loaded:");
    info!("  Training samples: {}", pipeline.num_train_samples());
    info!("  Validation samples: {}", pipeline.num_val_samples());
    info!("  Labels (Sell/Hold/Buy): {}/{}/{}", sell, hold, buy);

    Ok(pipeline)
}

/// Generate synthetic market data for testing
fn generate_synthetic_data(n: usize) -> Vec<Candle> {
    use rand::RngExt;
    let mut rng = rand::rng();

    let mut price = 50000.0;
    (0..n)
        .map(|i| {
            price *= 1.0 + rng.random_range(-0.02..0.02);
            let volatility = 0.01;

            Candle {
                timestamp: i as i64 * 60,
                open: price * (1.0 + rng.random_range(-volatility / 2.0..volatility / 2.0)),
                high: price * (1.0 + rng.random_range(0.0..volatility)),
                low: price * (1.0 - rng.random_range(0.0..volatility)),
                close: price * (1.0 + rng.random_range(-volatility / 2.0..volatility / 2.0)),
                volume: rng.random_range(100.0..10000.0),
            }
        })
        .collect()
}

/// Distributed training loop
async fn train_distributed(
    coordinator: &Arc<MultiNodeCoordinator>,
    _data_pipeline: Option<MarketDataPipeline>,
    args: &Args,
) -> Result<()> {
    let start_time = Instant::now();

    info!("Starting distributed training for {} epochs", args.epochs);

    // Create checkpoint coordinator
    let checkpoint = MultiNodeCheckpoint::new(Arc::clone(coordinator));

    for epoch in 0..args.epochs {
        info!("\n=== Epoch {}/{} ===", epoch + 1, args.epochs);

        // Train one epoch
        train_epoch(coordinator, epoch, args).await?;

        // Validate (on master)
        if coordinator.is_master() {
            validate_epoch(coordinator, epoch).await?;
        }

        // Coordinated checkpoint save
        if (epoch + 1) % 5 == 0 {
            let checkpoint_name = format!("multinode_epoch_{}", epoch + 1);
            checkpoint.save(&checkpoint_name).await?;
        }

        // Check node health
        if !coordinator.all_healthy().await {
            let failed = coordinator.failed_nodes().await;
            warn!("Failed nodes detected: {:?}", failed);

            if args.world_size > 1 {
                warn!("Continuing with remaining nodes (fault tolerance)");
            }
        }

        // Barrier between epochs
        coordinator.barrier().await?;
    }

    let total_time = start_time.elapsed();

    if coordinator.is_master() {
        info!("\n=== Training Complete ===");
        info!("Total time: {:.1} minutes", total_time.as_secs_f64() / 60.0);

        // Print final metrics
        let metrics = coordinator.metrics().await;
        info!("\nDistributed Training Metrics:");
        info!("  Total steps: {}", metrics.total_steps);
        info!("  Avg network sync: {:.2}ms", metrics.avg_network_sync_ms);
        info!(
            "  Network bandwidth: {:.1} MB/s",
            metrics.network_bandwidth_mbps
        );
        info!(
            "  Total bytes sent: {} MB",
            metrics.total_bytes_sent / 1024 / 1024
        );
        info!("  Failed syncs: {}", metrics.failed_syncs);
    }

    Ok(())
}

/// Train for one epoch
async fn train_epoch(
    coordinator: &Arc<MultiNodeCoordinator>,
    epoch: usize,
    _args: &Args,
) -> Result<()> {
    let num_steps = 100; // Placeholder
    let mut epoch_loss = 0.0;

    for step in 0..num_steps {
        // Simulate gradient computation (in practice, from actual model)
        let local_gradients = generate_dummy_gradients(1024);

        // First: sync gradients within this node's GPUs (handled by local_coordinator)
        // Then: sync gradients across nodes
        let synced_gradients = coordinator
            .sync_gradients("model.layer1", local_gradients)
            .await?;

        // Simulate applying gradients
        let loss = synced_gradients.iter().sum::<f32>() / synced_gradients.len() as f32;
        epoch_loss += loss as f64;

        coordinator.increment_step().await;

        // Log progress
        if step % 20 == 0 && coordinator.is_master() {
            info!(
                "  Rank {}, Epoch {}, Step {}/{}: loss={:.4}",
                coordinator.rank(),
                epoch + 1,
                step + 1,
                num_steps,
                loss
            );
        }
    }

    if coordinator.is_master() {
        info!(
            "Epoch {} complete: avg_loss={:.4}",
            epoch + 1,
            epoch_loss / num_steps as f64
        );
    }

    Ok(())
}

/// Validate for one epoch (master only)
async fn validate_epoch(coordinator: &Arc<MultiNodeCoordinator>, epoch: usize) -> Result<()> {
    info!("Running validation (rank {})...", coordinator.rank());

    let num_val_steps = 20;
    let mut val_loss = 0.0;

    for _ in 0..num_val_steps {
        // Simulate validation
        let dummy_loss = 0.5 + (epoch as f32 * -0.02);
        val_loss += dummy_loss as f64;
    }

    let avg_val_loss = val_loss / num_val_steps as f64;
    info!("Validation: loss={:.4}", avg_val_loss);

    Ok(())
}

/// Generate dummy gradients for demonstration
fn generate_dummy_gradients(size: usize) -> Vec<f32> {
    use rand::RngExt;
    let mut rng = rand::rng();
    (0..size).map(|_| rng.random_range(-0.1..0.1)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_args_default() {
        let args = Args::default();
        assert_eq!(args.rank, 0);
        assert_eq!(args.world_size, 1);
    }

    #[test]
    fn test_args_validation() {
        let args = Args {
            rank: 0,
            world_size: 2,
            node_addrs: vec!["localhost:50051".to_string(), "localhost:50052".to_string()],
            ..Default::default()
        };
        assert!(args.validate().is_ok());

        let bad_args = Args {
            rank: 2,
            world_size: 2,
            ..Default::default()
        };
        assert!(bad_args.validate().is_err());
    }

    #[test]
    fn test_generate_synthetic_data() {
        let data = generate_synthetic_data(100);
        assert_eq!(data.len(), 100);

        for candle in &data {
            assert!(candle.high >= candle.low);
            assert!(candle.high >= candle.open);
            assert!(candle.high >= candle.close);
        }
    }

    #[test]
    fn test_generate_dummy_gradients() {
        let grads = generate_dummy_gradients(100);
        assert_eq!(grads.len(), 100);
        assert!(grads.iter().all(|&g| g >= -0.1 && g <= 0.1));
    }
}
