//! gRPC Client for Distributed Training
//!
//! This module implements the gRPC client for distributed training communication.
//! It provides methods to push/pull gradients, synchronize barriers, and monitor health.
//!
//! # Example
//!
//! ```ignore
//! use janus_neuromorphic::distributed::grpc_client::DistributedTrainingClient;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let client = DistributedTrainingClient::connect("http://localhost:50051").await?;
//!
//! // Push gradients
//! let gradients = vec![1.0, 2.0, 3.0];
//! client.push_gradients("layer1", gradients, 0, 0).await?;
//!
//! // Pull parameters
//! let params = client.pull_parameters("layer1", 0).await?;
//! # Ok(())
//! # }
//! ```

use anyhow::{Result, anyhow};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tonic::transport::Channel;
use tracing::{debug, info, warn};

// Import from centralized fks-proto crate
pub use fks_proto::neuromorphic::distributed::*;

// Re-export the client type
use distributed_training_service_client::DistributedTrainingServiceClient as TonicClient;

/// Configuration for gRPC client
#[derive(Debug, Clone)]
pub struct ClientConfig {
    /// Server address
    pub server_addr: String,
    /// Connection timeout
    pub connect_timeout: Duration,
    /// Request timeout
    pub request_timeout: Duration,
    /// Enable keepalive
    pub keepalive: bool,
    /// Keepalive interval
    pub keepalive_interval: Duration,
    /// Max retry attempts
    pub max_retries: usize,
    /// Retry delay
    pub retry_delay: Duration,
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self {
            server_addr: "http://localhost:50051".to_string(),
            connect_timeout: Duration::from_secs(10),
            request_timeout: Duration::from_secs(30),
            keepalive: true,
            keepalive_interval: Duration::from_secs(10),
            max_retries: 3,
            retry_delay: Duration::from_millis(100),
        }
    }
}

/// Distributed training gRPC client
#[derive(Clone)]
pub struct DistributedTrainingClient {
    /// gRPC client
    client: TonicClient<Channel>,
    /// Client configuration
    #[allow(dead_code)]
    config: ClientConfig,
    /// Client rank
    rank: u32,
}

impl DistributedTrainingClient {
    /// Connect to a distributed training server
    pub async fn connect(addr: impl Into<String>) -> Result<Self> {
        let config = ClientConfig {
            server_addr: addr.into(),
            ..Default::default()
        };
        Self::with_config(config).await
    }

    /// Connect with custom configuration
    pub async fn with_config(config: ClientConfig) -> Result<Self> {
        info!(
            "Connecting to distributed training server: {}",
            config.server_addr
        );

        let endpoint = Channel::from_shared(config.server_addr.clone())
            .map_err(|e| anyhow!("Invalid server address: {}", e))?
            .connect_timeout(config.connect_timeout)
            .timeout(config.request_timeout);

        let endpoint = if config.keepalive {
            endpoint
                .http2_keep_alive_interval(config.keepalive_interval)
                .keep_alive_while_idle(true)
        } else {
            endpoint
        };

        let channel = endpoint
            .connect()
            .await
            .map_err(|e| anyhow!("Failed to connect: {}", e))?;

        let client = TonicClient::new(channel);

        Ok(Self {
            client,
            config,
            rank: 0,
        })
    }

    /// Set the client's rank
    pub fn set_rank(&mut self, rank: u32) {
        self.rank = rank;
    }

    /// Get the client's rank
    pub fn rank(&self) -> u32 {
        self.rank
    }

    /// Push gradients to parameter server
    pub async fn push_gradients(
        &mut self,
        key: impl Into<String>,
        data: Vec<f32>,
        version: u64,
        rank: u32,
    ) -> Result<PushGradientsResponse> {
        let key = key.into();
        debug!(
            "Pushing gradients for key={}, rank={}, size={}",
            key,
            rank,
            data.len()
        );

        let request = PushGradientsRequest {
            key: key.clone(),
            data,
            version,
            rank,
            shape: None,
            compression: 0, // NONE
            compressed_data: vec![],
        };

        let response = self
            .client
            .push_gradients(request)
            .await
            .map(|r| r.into_inner())
            .map_err(|e| anyhow::anyhow!("Push gradients failed: {}", e))?;

        if !response.success {
            warn!(
                "Push gradients failed for key={}: {}",
                key, response.message
            );
        }

        Ok(response)
    }

    /// Push gradients with shape information
    pub async fn push_gradients_with_shape(
        &mut self,
        key: impl Into<String>,
        data: Vec<f32>,
        shape: Vec<u64>,
        version: u64,
        rank: u32,
    ) -> Result<PushGradientsResponse> {
        let key = key.into();
        debug!(
            "Pushing gradients for key={}, rank={}, shape={:?}",
            key, rank, shape
        );

        let request = PushGradientsRequest {
            key: key.clone(),
            data,
            version,
            rank,
            shape: Some(TensorShape { dims: shape }),
            compression: 0, // NONE
            compressed_data: vec![],
        };

        let response = self
            .client
            .push_gradients(request)
            .await
            .map(|r| r.into_inner())
            .map_err(|e| anyhow::anyhow!("Push gradients failed: {}", e))?;

        if !response.success {
            warn!(
                "Push gradients failed for key={}: {}",
                key, response.message
            );
        }

        Ok(response)
    }

    /// Pull parameters from parameter server
    pub async fn pull_parameters(
        &mut self,
        key: impl Into<String>,
        rank: u32,
    ) -> Result<PullParametersResponse> {
        let key = key.into();
        debug!("Pulling parameters for key={}, rank={}", key, rank);

        let request = PullParametersRequest {
            key: key.clone(),
            version: None,
            rank,
        };

        let response = self
            .client
            .pull_parameters(request)
            .await
            .map(|r| r.into_inner())
            .map_err(|e| anyhow::anyhow!("Pull parameters failed: {}", e))?;

        if !response.success {
            return Err(anyhow!(
                "Pull parameters failed for key={}: {}",
                key,
                response.message
            ));
        }

        Ok(response)
    }

    /// Wait at a barrier
    pub async fn barrier(&mut self, barrier_id: u64, rank: u32) -> Result<BarrierResponse> {
        debug!("Waiting at barrier_id={}, rank={}", barrier_id, rank);

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let request = BarrierRequest {
            rank,
            barrier_id,
            timestamp,
        };

        // Keep retrying until barrier is released
        loop {
            let response = self
                .client
                .barrier(request.clone())
                .await
                .map(|r| r.into_inner())
                .map_err(|e| anyhow!("Barrier request failed: {}", e))?;

            if response.success {
                debug!("Barrier released: barrier_id={}", barrier_id);
                return Ok(response);
            }

            // Wait a bit before retrying
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    /// Send heartbeat
    pub async fn heartbeat(
        &mut self,
        rank: u32,
        status: i32,
        resources: Option<ResourceUsage>,
    ) -> Result<HeartbeatResponse> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let node_status = Some(NodeStatus {
            status,
            message: String::new(),
            last_update: timestamp,
        });

        let request = HeartbeatRequest {
            rank,
            timestamp,
            status: node_status,
            resources,
        };

        let response = self
            .client
            .heartbeat(request)
            .await
            .map(|r| r.into_inner())
            .map_err(|e| anyhow!("Heartbeat failed: {}", e))?;

        Ok(response)
    }

    /// All-reduce operation
    pub async fn all_reduce(
        &mut self,
        key: impl Into<String>,
        data: Vec<f32>,
        op: i32, // ReduceOp enum value
        rank: u32,
    ) -> Result<AllReduceResponse> {
        let key = key.into();
        debug!("All-reduce for key={}, rank={}, op={}", key, rank, op);

        let request = AllReduceRequest {
            key: key.clone(),
            data,
            op,
            rank,
            shape: None,
        };

        let response = self
            .client
            .all_reduce(request)
            .await
            .map(|r| r.into_inner())
            .map_err(|e| anyhow::anyhow!("All reduce failed: {}", e))?;

        if !response.success {
            return Err(anyhow!(
                "All-reduce failed for key={}: {}",
                key,
                response.message
            ));
        }

        Ok(response)
    }

    /// Broadcast operation
    pub async fn broadcast(
        &mut self,
        key: impl Into<String>,
        data: Vec<f32>,
        root: u32,
        rank: u32,
    ) -> Result<BroadcastResponse> {
        let key = key.into();
        debug!("Broadcast for key={}, root={}, rank={}", key, root, rank);

        let request = BroadcastRequest {
            key: key.clone(),
            data,
            root,
            rank,
            shape: None,
        };

        let response = self
            .client
            .broadcast(request)
            .await
            .map(|r| r.into_inner())
            .map_err(|e| anyhow::anyhow!("Broadcast failed: {}", e))?;

        if !response.success {
            return Err(anyhow!(
                "Broadcast failed for key={}: {}",
                key,
                response.message
            ));
        }

        Ok(response)
    }

    /// Gather operation
    pub async fn gather(
        &mut self,
        key: impl Into<String>,
        data: Vec<f32>,
        root: u32,
        rank: u32,
    ) -> Result<GatherResponse> {
        let key = key.into();
        debug!("Gather for key={}, root={}, rank={}", key, root, rank);

        let request = GatherRequest {
            key,
            data,
            root,
            rank,
            shape: None,
        };

        let response = self
            .client
            .gather(request)
            .await
            .map(|r| r.into_inner())
            .map_err(|e| anyhow::anyhow!("Gather failed: {}", e))?;

        if !response.success {
            return Err(anyhow!("Gather failed: {}", response.message));
        }

        Ok(response)
    }

    /// Register node with the cluster
    pub async fn register_node(
        &mut self,
        rank: u32,
        hostname: String,
        ip_address: String,
        gpu_ids: Vec<u32>,
        capabilities: NodeCapabilities,
    ) -> Result<RegisterNodeResponse> {
        info!(
            "Registering node: rank={}, hostname={}, ip={}",
            rank, hostname, ip_address
        );

        let request = RegisterNodeRequest {
            rank,
            hostname,
            ip_address,
            gpu_ids,
            capabilities: Some(capabilities),
        };

        let response = self
            .client
            .register_node(request)
            .await
            .map(|r| r.into_inner())
            .map_err(|e| anyhow::anyhow!("Register node failed: {}", e))?;

        if !response.success {
            return Err(anyhow!("Node registration failed: {}", response.message));
        }

        info!(
            "Node registered successfully: assigned_rank={}, world_size={}",
            response.assigned_rank, response.world_size
        );

        Ok(response)
    }

    /// Get cluster status
    pub async fn get_cluster_status(&mut self, rank: u32) -> Result<GetClusterStatusResponse> {
        debug!("Getting cluster status for rank={}", rank);

        let request = GetClusterStatusRequest { rank };

        let response = self
            .client
            .get_cluster_status(request)
            .await
            .map(|r| r.into_inner())
            .map_err(|e| anyhow!("Get cluster status failed: {}", e))?;

        Ok(response)
    }
}

/// Helper functions for creating common request types
/// Create node capabilities
pub fn create_node_capabilities(
    has_cuda: bool,
    has_nccl: bool,
    num_gpus: u32,
    gpu_type: String,
) -> NodeCapabilities {
    NodeCapabilities {
        has_cuda,
        has_nccl,
        has_rdma: false,
        num_gpus,
        gpu_type,
        total_memory: 0,
        num_cpus: num_cpus::get() as u32,
    }
}

/// Create resource usage info
pub fn create_resource_usage() -> ResourceUsage {
    ResourceUsage {
        cpu_usage: 0.0,
        memory_usage: 0.0,
        gpu_usage: vec![],
        network_bandwidth: 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_config_default() {
        let config = ClientConfig::default();
        assert_eq!(config.server_addr, "http://localhost:50051");
        assert_eq!(config.max_retries, 3);
    }

    #[test]
    fn test_create_node_capabilities() {
        let caps = create_node_capabilities(true, true, 4, "NVIDIA A100".to_string());
        assert!(caps.has_cuda);
        assert!(caps.has_nccl);
        assert_eq!(caps.num_gpus, 4);
    }
}
