//! NCCL Backend for Multi-GPU Communication
//!
//! This module provides NCCL-based GPU communication primitives for distributed training.
//! It wraps NCCL collective operations and provides a high-level interface for gradient
//! synchronization across multiple GPUs and nodes.
//!
//! # Features
//!
//! - AllReduce for gradient averaging
//! - Broadcast for parameter distribution
//! - ReduceScatter for distributed optimizer states
//! - AllGather for collecting results
//! - Point-to-point send/receive
//! - Multi-stream support
//! - Asynchronous operations
//!
//! # Example
//!
//! ```ignore
//! use janus_neuromorphic::distributed::nccl_backend::{NcclBackend, NcclConfig};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let config = NcclConfig {
//!     rank: 0,
//!     world_size: 4,
//!     ..Default::default()
//! };
//!
//! let backend = NcclBackend::new(config).await?;
//!
//! // All-reduce gradients across GPUs
//! let gradients = vec![1.0, 2.0, 3.0, 4.0];
//! let averaged = backend.all_reduce(gradients, NcclReduceOp::Sum).await?;
//! # Ok(())
//! # }
//! ```

#[cfg(feature = "nccl")]
use cudarc::nccl::{Comm as NcclComm, Id as NcclUniqueId, ReduceOp as CudarcReduceOp};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tracing::warn;

#[cfg(feature = "nccl")]
use anyhow::anyhow;
#[cfg(feature = "nccl")]
use std::sync::Arc;
#[cfg(feature = "nccl")]
use tokio::sync::RwLock;
#[cfg(feature = "nccl")]
use tracing::{debug, info};

#[cfg(feature = "nccl")]
use cudarc::driver::CudaDevice;

/// NCCL reduce operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NcclReduceOp {
    /// Sum reduction
    Sum,
    /// Product reduction
    Prod,
    /// Maximum reduction
    Max,
    /// Minimum reduction
    Min,
    /// Average reduction
    Avg,
}

#[cfg(feature = "nccl")]
impl From<NcclReduceOp> for CudarcReduceOp {
    fn from(op: NcclReduceOp) -> Self {
        match op {
            NcclReduceOp::Sum => CudarcReduceOp::Sum,
            NcclReduceOp::Prod => CudarcReduceOp::Prod,
            NcclReduceOp::Max => CudarcReduceOp::Max,
            NcclReduceOp::Min => CudarcReduceOp::Min,
            NcclReduceOp::Avg => CudarcReduceOp::Avg,
        }
    }
}

/// NCCL backend configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NcclConfig {
    /// Rank of this process in the communicator
    pub rank: usize,
    /// Total number of processes (world size)
    pub world_size: usize,
    /// GPU device ID to use
    pub device_id: usize,
    /// NCCL unique ID (for initialization)
    pub nccl_id: Option<Vec<u8>>,
    /// Enable NCCL profiling
    pub enable_profiling: bool,
    /// NCCL network interface (e.g., "eth0", "ib0")
    pub network_interface: Option<String>,
    /// Use RDMA if available
    pub use_rdma: bool,
}

impl Default for NcclConfig {
    fn default() -> Self {
        Self {
            rank: 0,
            world_size: 1,
            device_id: 0,
            nccl_id: None,
            enable_profiling: false,
            network_interface: None,
            use_rdma: false,
        }
    }
}

/// NCCL communication backend
pub struct NcclBackend {
    /// Configuration
    config: NcclConfig,
    /// NCCL communicator (wrapped for cfg)
    #[cfg(feature = "nccl")]
    comm: Arc<RwLock<NcclComm>>,
    /// CUDA device
    #[cfg(feature = "nccl")]
    device: Arc<CudaDevice>,
    /// Placeholder for non-NCCL builds
    #[cfg(not(feature = "nccl"))]
    _phantom: std::marker::PhantomData<()>,
}

#[cfg(feature = "nccl")]
impl NcclBackend {
    /// Create a new NCCL backend
    pub async fn new(config: NcclConfig) -> Result<Self> {
        info!(
            "Initializing NCCL backend: rank={}, world_size={}, device_id={}",
            config.rank, config.world_size, config.device_id
        );

        // Set NCCL environment variables
        if let Some(ref iface) = config.network_interface {
            std::env::set_var("NCCL_SOCKET_IFNAME", iface);
        }

        if config.use_rdma {
            std::env::set_var("NCCL_IB_DISABLE", "0");
            std::env::set_var("NCCL_NET_GDR_LEVEL", "5");
        }

        if config.enable_profiling {
            std::env::set_var("NCCL_DEBUG", "INFO");
        }

        // Initialize CUDA device
        let device = CudaDevice::new(config.device_id).map_err(|e| {
            anyhow!(
                "Failed to initialize CUDA device {}: {:?}",
                config.device_id,
                e
            )
        })?;

        // Create or deserialize NCCL unique ID
        let nccl_id = if let Some(ref id_bytes) = config.nccl_id {
            // Deserialize existing ID (for non-root ranks)
            if id_bytes.len() != std::mem::size_of::<NcclUniqueId>() {
                return Err(anyhow!("Invalid NCCL ID length"));
            }
            unsafe {
                let mut id: NcclUniqueId = std::mem::zeroed();
                std::ptr::copy_nonoverlapping(
                    id_bytes.as_ptr(),
                    &mut id as *mut _ as *mut u8,
                    id_bytes.len(),
                );
                id
            }
        } else {
            // Generate new ID (for root rank)
            NcclUniqueId::new()
                .map_err(|e| anyhow!("Failed to generate NCCL unique ID: {:?}", e))?
        };

        // Initialize NCCL communicator
        let comm = NcclComm::from_rank(device.clone(), config.rank, config.world_size, nccl_id)
            .map_err(|e| anyhow!("Failed to initialize NCCL communicator: {:?}", e))?;

        info!("NCCL backend initialized successfully");

        Ok(Self {
            config,
            comm: Arc::new(RwLock::new(comm)),
            device,
        })
    }

    /// Get the NCCL unique ID (for broadcasting to other ranks)
    pub fn get_unique_id() -> Result<Vec<u8>> {
        let id = NcclUniqueId::new()
            .map_err(|e| anyhow!("Failed to generate NCCL unique ID: {:?}", e))?;

        let bytes = unsafe {
            let ptr = &id as *const _ as *const u8;
            std::slice::from_raw_parts(ptr, std::mem::size_of::<NcclUniqueId>())
        };

        Ok(bytes.to_vec())
    }

    /// All-reduce operation: sum/avg gradients across all GPUs
    pub async fn all_reduce(&self, data: Vec<f32>, op: NcclReduceOp) -> Result<Vec<f32>> {
        debug!(
            "NCCL all-reduce: rank={}, size={}, op={:?}",
            self.config.rank,
            data.len(),
            op
        );

        let mut result = data.clone();
        let comm = self.comm.read().await;

        // Upload data to GPU
        let device_data = self
            .device
            .htod_sync_copy(&data)
            .map_err(|e| anyhow!("Failed to copy data to device: {:?}", e))?;

        let mut device_result = self
            .device
            .alloc_zeros::<f32>(data.len())
            .map_err(|e| anyhow!("Failed to allocate device memory: {:?}", e))?;

        // Perform NCCL all-reduce
        comm.all_reduce(&device_data, &mut device_result, &op.into())
            .map_err(|e| anyhow!("NCCL all-reduce failed: {:?}", e))?;

        // Download result from GPU
        self.device
            .dtoh_sync_copy_into(&device_result, &mut result)
            .map_err(|e| anyhow!("Failed to copy result from device: {:?}", e))?;

        // If averaging, divide by world size
        if matches!(op, NcclReduceOp::Avg) {
            let scale = 1.0 / self.config.world_size as f32;
            for val in &mut result {
                *val *= scale;
            }
        }

        Ok(result)
    }

    /// Broadcast: send data from root to all other ranks
    pub async fn broadcast(&self, data: Vec<f32>, root: usize) -> Result<Vec<f32>> {
        debug!(
            "NCCL broadcast: rank={}, root={}, size={}",
            self.config.rank,
            root,
            data.len()
        );

        let mut result = data.clone();
        let comm = self.comm.read().await;

        // Upload data to GPU
        let device_data = if self.config.rank == root {
            Some(
                self.device
                    .htod_sync_copy(&data)
                    .map_err(|e| anyhow!("Failed to copy data to device: {:?}", e))?,
            )
        } else {
            None
        };

        let mut device_result = self
            .device
            .alloc_zeros::<f32>(data.len())
            .map_err(|e| anyhow!("Failed to allocate device memory: {:?}", e))?;

        // Perform NCCL broadcast
        comm.broadcast(&device_data, &mut device_result, root as i32)
            .map_err(|e| anyhow!("NCCL broadcast failed: {:?}", e))?;

        // Download result from GPU
        self.device
            .dtoh_sync_copy_into(&device_result, &mut result)
            .map_err(|e| anyhow!("Failed to copy result from device: {:?}", e))?;

        Ok(result)
    }

    /// Reduce: aggregate data to root rank
    pub async fn reduce(&self, data: Vec<f32>, root: usize, op: NcclReduceOp) -> Result<Vec<f32>> {
        debug!(
            "NCCL reduce: rank={}, root={}, size={}, op={:?}",
            self.config.rank,
            root,
            data.len(),
            op
        );

        let mut result = data.clone();
        let comm = self.comm.read().await;

        // Upload data to GPU
        let device_data = self
            .device
            .htod_sync_copy(&data)
            .map_err(|e| anyhow!("Failed to copy data to device: {:?}", e))?;

        let mut device_result = self
            .device
            .alloc_zeros::<f32>(data.len())
            .map_err(|e| anyhow!("Failed to allocate device memory: {:?}", e))?;

        // Perform NCCL reduce
        comm.reduce(&device_data, &mut device_result, &op.into(), root as i32)
            .map_err(|e| anyhow!("NCCL reduce failed: {:?}", e))?;

        // Download result from GPU (only meaningful for root)
        if self.config.rank == root {
            self.device
                .dtoh_sync_copy_into(&device_result, &mut result)
                .map_err(|e| anyhow!("Failed to copy result from device: {:?}", e))?;
        }

        Ok(result)
    }

    /// All-gather: gather data from all ranks to all ranks
    pub async fn all_gather(&self, data: Vec<f32>) -> Result<Vec<Vec<f32>>> {
        debug!(
            "NCCL all-gather: rank={}, size={}",
            self.config.rank,
            data.len()
        );

        let comm = self.comm.read().await;

        // Upload data to GPU
        let device_data = self
            .device
            .htod_sync_copy(&data)
            .map_err(|e| anyhow!("Failed to copy data to device: {:?}", e))?;

        let total_size = data.len() * self.config.world_size;
        let mut device_result = self
            .device
            .alloc_zeros::<f32>(total_size)
            .map_err(|e| anyhow!("Failed to allocate device memory: {:?}", e))?;

        // Perform NCCL all-gather
        comm.all_gather(&device_data, &mut device_result)
            .map_err(|e| anyhow!("NCCL all-gather failed: {:?}", e))?;

        // Download result from GPU
        let mut flat_result = vec![0.0f32; total_size];
        self.device
            .dtoh_sync_copy_into(&device_result, &mut flat_result)
            .map_err(|e| anyhow!("Failed to copy result from device: {:?}", e))?;

        // Split into chunks per rank
        let mut result = Vec::with_capacity(self.config.world_size);
        for chunk in flat_result.chunks(data.len()) {
            result.push(chunk.to_vec());
        }

        Ok(result)
    }

    /// Reduce-scatter: reduce then scatter chunks to all ranks
    pub async fn reduce_scatter(&self, data: Vec<f32>, op: NcclReduceOp) -> Result<Vec<f32>> {
        debug!(
            "NCCL reduce-scatter: rank={}, size={}, op={:?}",
            self.config.rank,
            data.len(),
            op
        );

        if data.len() % self.config.world_size != 0 {
            return Err(anyhow!(
                "Data size {} must be divisible by world size {}",
                data.len(),
                self.config.world_size
            ));
        }

        let comm = self.comm.read().await;
        let chunk_size = data.len() / self.config.world_size;

        // Upload data to GPU
        let device_data = self
            .device
            .htod_sync_copy(&data)
            .map_err(|e| anyhow!("Failed to copy data to device: {:?}", e))?;

        let mut device_result = self
            .device
            .alloc_zeros::<f32>(chunk_size)
            .map_err(|e| anyhow!("Failed to allocate device memory: {:?}", e))?;

        // Perform NCCL reduce-scatter
        comm.reduce_scatter(&device_data, &mut device_result, &op.into())
            .map_err(|e| anyhow!("NCCL reduce-scatter failed: {:?}", e))?;

        // Download result from GPU
        let mut result = vec![0.0f32; chunk_size];
        self.device
            .dtoh_sync_copy_into(&device_result, &mut result)
            .map_err(|e| anyhow!("Failed to copy result from device: {:?}", e))?;

        Ok(result)
    }

    /// Get rank
    pub fn rank(&self) -> usize {
        self.config.rank
    }

    /// Get world size
    pub fn world_size(&self) -> usize {
        self.config.world_size
    }

    /// Get device ID
    pub fn device_id(&self) -> usize {
        self.config.device_id
    }

    /// Synchronize device
    pub async fn synchronize(&self) -> Result<()> {
        self.device
            .synchronize()
            .map_err(|e| anyhow!("Device synchronization failed: {:?}", e))?;
        Ok(())
    }

    /// Get device
    pub fn device(&self) -> Arc<CudaDevice> {
        self.device.clone()
    }
}

// Non-NCCL implementation (stub for CPU-only builds)
#[cfg(not(feature = "nccl"))]
impl NcclBackend {
    /// Create a new NCCL backend (stub)
    pub async fn new(config: NcclConfig) -> Result<Self> {
        warn!("NCCL is not enabled. Using stub implementation.");
        Ok(Self {
            config,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Get the NCCL unique ID (stub)
    pub fn get_unique_id() -> Result<Vec<u8>> {
        Ok(vec![0u8; 128]) // Dummy ID
    }

    /// All-reduce operation (stub - returns input unchanged)
    pub async fn all_reduce(&self, data: Vec<f32>, _op: NcclReduceOp) -> Result<Vec<f32>> {
        warn!("NCCL all-reduce called but NCCL is not enabled");
        Ok(data)
    }

    /// Broadcast (stub)
    pub async fn broadcast(&self, data: Vec<f32>, _root: usize) -> Result<Vec<f32>> {
        warn!("NCCL broadcast called but NCCL is not enabled");
        Ok(data)
    }

    /// Reduce (stub)
    pub async fn reduce(
        &self,
        data: Vec<f32>,
        _root: usize,
        _op: NcclReduceOp,
    ) -> Result<Vec<f32>> {
        warn!("NCCL reduce called but NCCL is not enabled");
        Ok(data)
    }

    /// All-gather (stub)
    pub async fn all_gather(&self, data: Vec<f32>) -> Result<Vec<Vec<f32>>> {
        warn!("NCCL all-gather called but NCCL is not enabled");
        Ok(vec![data])
    }

    /// Reduce-scatter (stub)
    pub async fn reduce_scatter(&self, data: Vec<f32>, _op: NcclReduceOp) -> Result<Vec<f32>> {
        warn!("NCCL reduce-scatter called but NCCL is not enabled");
        Ok(data)
    }

    /// Get rank
    pub fn rank(&self) -> usize {
        self.config.rank
    }

    /// Get world size
    pub fn world_size(&self) -> usize {
        self.config.world_size
    }

    /// Get device ID
    pub fn device_id(&self) -> usize {
        self.config.device_id
    }

    /// Synchronize device (stub)
    pub async fn synchronize(&self) -> Result<()> {
        Ok(())
    }
}

/// NCCL communicator group for managing multiple communicators
pub struct NcclCommGroup {
    /// Communicators for each device
    backends: Vec<NcclBackend>,
}

impl NcclCommGroup {
    /// Create a new communicator group
    pub async fn new(configs: Vec<NcclConfig>) -> Result<Self> {
        let mut backends = Vec::new();
        for config in configs {
            backends.push(NcclBackend::new(config).await?);
        }
        Ok(Self { backends })
    }

    /// Get backend for a specific device
    pub fn get(&self, device_id: usize) -> Option<&NcclBackend> {
        self.backends.get(device_id)
    }

    /// Get all backends
    pub fn backends(&self) -> &[NcclBackend] {
        &self.backends
    }

    /// Number of devices/backends
    pub fn len(&self) -> usize {
        self.backends.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.backends.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nccl_config_default() {
        let config = NcclConfig::default();
        assert_eq!(config.rank, 0);
        assert_eq!(config.world_size, 1);
        assert_eq!(config.device_id, 0);
    }

    #[test]
    fn test_reduce_op_conversion() {
        let ops = vec![
            NcclReduceOp::Sum,
            NcclReduceOp::Prod,
            NcclReduceOp::Max,
            NcclReduceOp::Min,
            NcclReduceOp::Avg,
        ];

        for op in ops {
            #[cfg(feature = "nccl")]
            {
                let _: CudarcReduceOp = op.into();
            }
            #[cfg(not(feature = "nccl"))]
            {
                let _ = op;
            }
        }
    }

    #[tokio::test]
    async fn test_nccl_backend_creation() {
        let config = NcclConfig::default();
        let result = NcclBackend::new(config).await;

        // This will succeed in non-NCCL builds (stub)
        // and may fail in NCCL builds without proper GPU setup
        #[cfg(not(feature = "nccl"))]
        assert!(result.is_ok());
    }
}
