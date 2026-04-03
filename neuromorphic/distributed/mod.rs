//! Distributed Training Infrastructure
//!
//! This module provides comprehensive distributed training support for the FKS neuromorphic system.
//! It enables multi-GPU and multi-node training with efficient gradient synchronization,
//! data parallelism, and checkpoint coordination.
//!
//! # Features
//!
//! - **Multi-GPU Training**: Automatic device detection and distributed coordination
//! - **Data Parallelism**: Efficient data sharding and distribution across devices
//! - **Gradient Synchronization**: Multiple sync strategies (AllReduce, ParameterServer, RingAllReduce)
//! - **Distributed Checkpoints**: Coordinated checkpoint management with cloud storage support
//! - **Performance Monitoring**: Real-time metrics and profiling for distributed training
//!
//! # Architecture
//!
//! The distributed training system consists of three main components:
//!
//! 1. **TrainingCoordinator**: Manages devices, gradient synchronization, and communication
//! 2. **DistributedDataLoader**: Handles data sharding, prefetching, and balanced distribution
//! 3. **DistributedCheckpointManager**: Coordinates checkpoint saving/loading across devices
//!
//! # Example: Basic Multi-GPU Training
//!
//! ```no_run
//! use janus_neuromorphic::distributed::{
//!     TrainingCoordinator, DistributedDataLoader, DistributedConfig, ShardingStrategy
//! };
//! use candle_core::Tensor;
//!
//! # fn example() -> anyhow::Result<()> {
//! // Initialize coordinator
//! let coordinator = TrainingCoordinator::new()?;
//! println!("Training on {} devices", coordinator.available_devices().len());
//!
//! // Create distributed data loader
//! let data: Vec<Vec<f32>> = vec![vec![1.0; 10]; 1000];
//! let mut loader = DistributedDataLoader::new(data, 32, coordinator.available_devices().len())?
//!     .with_sharding(ShardingStrategy::RoundRobin)
//!     .with_shuffle(true);
//!
//! // Training loop
//! for (batch_data, labels, _indices) in loader.iter() {
//!     // Forward pass on each device
//!     // Backward pass (gradients computed)
//!
//!     // Accumulate gradients
//!     for (param_name, gradient) in compute_gradients() {
//!         coordinator.accumulate_gradient(&param_name, gradient)?;
//!     }
//!
//!     // Synchronize gradients across devices
//!     if coordinator.should_sync() {
//!         let synced_grads = coordinator.sync_gradients()?;
//!         // Apply synced gradients to model
//!     }
//!
//!     coordinator.increment_step();
//! }
//!
//! # fn compute_gradients() -> Vec<(String, Tensor)> { vec![] }
//! # Ok(())
//! # }
//! ```
//!
//! # Example: Distributed Checkpoint Management
//!
//! ```no_run
//! use janus_neuromorphic::distributed::{
//!     DistributedCheckpointManager, CheckpointConfig, StorageBackend
//! };
//! use std::collections::HashMap;
//!
//! # fn example() -> anyhow::Result<()> {
//! // Create checkpoint manager
//! let mut manager = DistributedCheckpointManager::new("checkpoints")?;
//!
//! // Save checkpoint
//! let mut state = HashMap::new();
//! // state.insert("model_weights", tensors);
//! let metadata = manager.save_checkpoint("model_v1", state, 1000)?;
//! println!("Saved checkpoint at step {}", metadata.step);
//!
//! // Load checkpoint
//! let loaded_state = manager.load_checkpoint("model_v1_v1")?;
//! # Ok(())
//! # }
//! ```
//!
//! # Gradient Synchronization Strategies
//!
//! ## AllReduce (Default)
//! - Averages gradients across all devices
//! - Best for homogeneous GPU clusters
//! - Symmetric communication pattern
//!
//! ## Parameter Server
//! - Centralized gradient aggregation on rank 0
//! - Useful for heterogeneous setups
//! - Can become bottleneck at scale
//!
//! ## Ring AllReduce
//! - Bandwidth-optimal gradient synchronization
//! - Reduces communication overhead
//! - Best for large models and many GPUs
//!
//! # Data Sharding Strategies
//!
//! - **Contiguous**: Sequential chunks to each device
//! - **RoundRobin**: Alternating distribution (better load balancing)
//! - **Random**: Random sharding with reproducible seed
//! - **Stratified**: Preserves class distribution (for classification)
//!
//! # Performance Considerations
//!
//! - **Gradient Accumulation**: Reduce sync frequency for small batches
//! - **Prefetching**: Overlap data loading with computation
//! - **Mixed Precision**: Use fp16 for faster computation and reduced memory
//! - **Gradient Compression**: Reduce communication overhead (experimental)
//!
//! # Fault Tolerance
//!
//! - Automatic checkpoint rotation with configurable retention
//! - Metadata tracking for recovery
//! - Cloud storage backup for disaster recovery
//!
//! # See Also
//!
//! - [`TrainingCoordinator`] - Core distributed training coordination
//! - [`DistributedDataLoader`] - Data loading and sharding
//! - [`DistributedCheckpointManager`] - Checkpoint management

pub mod checkpoint;
pub mod coordinator;
pub mod data_loader;
pub mod multinode;
pub mod network;
pub mod runtime;

// gRPC communication
pub mod grpc_client;
pub mod grpc_server;

// NCCL backend for GPU communication
pub mod nccl_backend;

// Re-export main types
pub use checkpoint::{
    CheckpointConfig, CheckpointMetadata, DistributedCheckpointManager, ShardInfo, StorageBackend,
};
pub use coordinator::{
    DeviceInfo, DistributedConfig, DistributedMetrics, SyncMethod, TrainingCoordinator,
    TrainingStrategy,
};
pub use data_loader::{
    DataLoaderConfig, DistributedBatch, DistributedDataLoader, SamplingStrategy, ShardingStrategy,
    batches_to_tensors, labels_to_tensors,
};
pub use multinode::{MultiNodeCheckpoint, MultiNodeConfig, MultiNodeCoordinator, MultiNodeMetrics};
pub use network::{Message, NetworkBackend, NetworkConfig, ParameterServer};
pub use runtime::{DistributedRuntime, NodeInfo, NodeStatus, RuntimeConfig};

// gRPC types
pub use grpc_client::{ClientConfig as GrpcClientConfig, DistributedTrainingClient};
pub use grpc_server::DistributedTrainingServer;

// NCCL types
pub use nccl_backend::{NcclBackend, NcclCommGroup, NcclConfig, NcclReduceOp};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exports() {
        // Verify all main types are exported
        let _: Option<TrainingCoordinator> = None;
        let _: Option<DistributedDataLoader<Vec<f32>>> = None;
        let _: Option<DistributedCheckpointManager> = None;
        let _: Option<DistributedConfig> = None;
        let _: Option<CheckpointConfig> = None;
    }
}
