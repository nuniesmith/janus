//! gRPC Server for Distributed Training
//!
//! This module implements the gRPC server for the distributed training service.
//! It handles gradient synchronization, parameter serving, barriers, and health monitoring.
//!
//! # Example
//!
//! ```ignore
//! use janus_neuromorphic::distributed::grpc_server::DistributedTrainingServer;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let server = DistributedTrainingServer::new("0.0.0.0:50051").await?;
//! server.serve().await?;
//! # Ok(())
//! # }
//! ```

use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tonic::{Request, Response, Status, transport::Server};
use tracing::{debug, info};

// Import from centralized fks-proto crate
pub use fks_proto::neuromorphic::distributed::*;

use distributed_training_service_server::{
    DistributedTrainingService, DistributedTrainingServiceServer as TonicServer,
};

/// Parameter server state
#[derive(Debug, Clone)]
struct ParameterState {
    /// Current parameter values
    data: Vec<f32>,
    /// Parameter version
    version: u64,
    /// Shape information
    shape: Vec<u64>,
    /// Accumulated gradients (for averaging)
    accumulated_gradients: Vec<Vec<f32>>,
    /// Number of workers that have pushed gradients
    num_workers_pushed: usize,
}

/// Node information (internal struct)
#[derive(Debug, Clone)]
struct InternalNodeInfo {
    rank: u32,
    hostname: String,
    ip_address: String,
    status: i32,
    last_heartbeat: u64,
    capabilities: Option<NodeCapabilities>,
    resources: Option<ResourceUsage>,
}

/// Barrier state
#[derive(Debug, Clone)]
struct BarrierState {
    /// Barrier ID
    #[allow(dead_code)]
    id: u64,
    /// Ranks waiting at barrier
    waiting_ranks: Vec<u32>,
    /// Total expected ranks
    expected_ranks: usize,
}

/// Distributed training server implementation
pub struct DistributedTrainingServer {
    /// Server bind address
    addr: String,
    /// Parameter states (key -> state)
    parameters: Arc<RwLock<HashMap<String, ParameterState>>>,
    /// Registered nodes (rank -> info)
    nodes: Arc<RwLock<HashMap<u32, InternalNodeInfo>>>,
    /// Barrier states (barrier_id -> state)
    barriers: Arc<RwLock<HashMap<u64, BarrierState>>>,
    /// World size (total number of nodes)
    world_size: Arc<RwLock<u32>>,
    /// Master rank
    master_rank: u32,
}

impl DistributedTrainingServer {
    /// Create a new distributed training server
    pub fn new(addr: impl Into<String>) -> Self {
        Self {
            addr: addr.into(),
            parameters: Arc::new(RwLock::new(HashMap::new())),
            nodes: Arc::new(RwLock::new(HashMap::new())),
            barriers: Arc::new(RwLock::new(HashMap::new())),
            world_size: Arc::new(RwLock::new(0)),
            master_rank: 0,
        }
    }

    /// Start serving gRPC requests
    pub async fn serve(self) -> Result<()> {
        let addr = self.addr.parse()?;
        let service = DistributedTrainingServiceImpl {
            parameters: self.parameters.clone(),
            nodes: self.nodes.clone(),
            barriers: self.barriers.clone(),
            world_size: self.world_size.clone(),
            master_rank: self.master_rank,
        };

        info!("Starting distributed training gRPC server on {}", self.addr);

        Server::builder()
            .add_service(TonicServer::new(service))
            .serve(addr)
            .await?;

        Ok(())
    }

    /// Get current number of registered nodes
    pub async fn num_nodes(&self) -> usize {
        self.nodes.read().await.len()
    }

    /// Get world size
    pub async fn get_world_size(&self) -> u32 {
        *self.world_size.read().await
    }
}

/// gRPC service implementation
#[derive(Clone)]
struct DistributedTrainingServiceImpl {
    parameters: Arc<RwLock<HashMap<String, ParameterState>>>,
    nodes: Arc<RwLock<HashMap<u32, InternalNodeInfo>>>,
    barriers: Arc<RwLock<HashMap<u64, BarrierState>>>,
    world_size: Arc<RwLock<u32>>,
    master_rank: u32,
}

#[tonic::async_trait]
impl DistributedTrainingService for DistributedTrainingServiceImpl {
    async fn push_gradients(
        &self,
        request: Request<PushGradientsRequest>,
    ) -> Result<Response<PushGradientsResponse>, Status> {
        let req = request.into_inner();
        debug!(
            "Received push_gradients for key={}, rank={}, version={}",
            req.key, req.rank, req.version
        );

        let mut params = self.parameters.write().await;
        let state = params.entry(req.key.clone()).or_insert_with(|| {
            let shape = req
                .shape
                .as_ref()
                .map(|s| s.dims.clone())
                .unwrap_or_default();
            ParameterState {
                data: vec![0.0; req.data.len()],
                version: 0,
                shape,
                accumulated_gradients: Vec::new(),
                num_workers_pushed: 0,
            }
        });

        // Accumulate gradients
        state.accumulated_gradients.push(req.data.clone());
        state.num_workers_pushed += 1;

        // If all workers have pushed, average gradients and update parameters
        let world_size = *self.world_size.read().await as usize;
        if state.num_workers_pushed >= world_size {
            debug!(
                "All workers pushed gradients for key={}, averaging",
                req.key
            );

            // Average accumulated gradients
            let num_grads = state.accumulated_gradients.len();
            let grad_size = state.accumulated_gradients[0].len();
            let mut averaged = vec![0.0; grad_size];

            for grad in &state.accumulated_gradients {
                for (i, &val) in grad.iter().enumerate() {
                    averaged[i] += val;
                }
            }

            for val in &mut averaged {
                *val /= num_grads as f32;
            }

            // Update parameters (apply gradients)
            for (i, grad) in averaged.iter().enumerate() {
                state.data[i] -= grad; // Simple SGD step
            }

            state.version += 1;
            state.accumulated_gradients.clear();
            state.num_workers_pushed = 0;
        }

        Ok(Response::new(PushGradientsResponse {
            success: true,
            message: "Gradients pushed successfully".to_string(),
            version: state.version,
        }))
    }

    async fn pull_parameters(
        &self,
        request: Request<PullParametersRequest>,
    ) -> Result<Response<PullParametersResponse>, Status> {
        let req = request.into_inner();
        debug!(
            "Received pull_parameters for key={}, rank={}",
            req.key, req.rank
        );

        let params = self.parameters.read().await;

        if let Some(state) = params.get(&req.key) {
            let shape = Some(TensorShape {
                dims: state.shape.clone(),
            });

            Ok(Response::new(PullParametersResponse {
                success: true,
                message: "Parameters retrieved successfully".to_string(),
                data: state.data.clone(),
                version: state.version,
                shape,
            }))
        } else {
            Ok(Response::new(PullParametersResponse {
                success: false,
                message: format!("Parameter {} not found", req.key),
                data: vec![],
                version: 0,
                shape: None,
            }))
        }
    }

    async fn barrier(
        &self,
        request: Request<BarrierRequest>,
    ) -> Result<Response<BarrierResponse>, Status> {
        let req = request.into_inner();
        debug!(
            "Received barrier from rank={}, barrier_id={}",
            req.rank, req.barrier_id
        );

        let mut barriers = self.barriers.write().await;
        let world_size = *self.world_size.read().await as usize;

        let barrier = barriers
            .entry(req.barrier_id)
            .or_insert_with(|| BarrierState {
                id: req.barrier_id,
                waiting_ranks: Vec::new(),
                expected_ranks: world_size,
            });

        // Add rank to waiting list if not already there
        if !barrier.waiting_ranks.contains(&req.rank) {
            barrier.waiting_ranks.push(req.rank);
        }

        let num_waiting = barrier.waiting_ranks.len();
        let expected_ranks = barrier.expected_ranks;
        let success = num_waiting >= expected_ranks;

        // Clear barrier if all ranks arrived
        if success {
            barriers.remove(&req.barrier_id);
        }

        Ok(Response::new(BarrierResponse {
            success,
            message: if success {
                "Barrier released".to_string()
            } else {
                format!("Waiting for {} more ranks", expected_ranks - num_waiting)
            },
            barrier_id: req.barrier_id,
            num_waiting: num_waiting as u32,
        }))
    }

    async fn heartbeat(
        &self,
        request: Request<HeartbeatRequest>,
    ) -> Result<Response<HeartbeatResponse>, Status> {
        let req = request.into_inner();
        debug!("Received heartbeat from rank={}", req.rank);

        let mut nodes = self.nodes.write().await;

        if let Some(node) = nodes.get_mut(&req.rank) {
            node.last_heartbeat = req.timestamp;
            node.status = req.status.map(|s| s.status).unwrap_or(0);
            node.resources = req.resources;
        }

        // Calculate cluster health
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let num_healthy = nodes
            .values()
            .filter(|n| now - n.last_heartbeat < 30000) // 30s timeout
            .count();

        let num_failed = nodes.len() - num_healthy;

        let health = if num_failed == 0 {
            1 // HEALTHY
        } else if num_failed < nodes.len() / 2 {
            2 // DEGRADED
        } else {
            3 // UNHEALTHY
        };

        let cluster_health = Some(ClusterHealth {
            health,
            num_healthy_nodes: num_healthy as u32,
            num_failed_nodes: num_failed as u32,
            message: format!("{}/{} nodes healthy", num_healthy, nodes.len()),
        });

        Ok(Response::new(HeartbeatResponse {
            success: true,
            message: "Heartbeat acknowledged".to_string(),
            cluster_health,
        }))
    }

    async fn all_reduce(
        &self,
        request: Request<AllReduceRequest>,
    ) -> Result<Response<AllReduceResponse>, Status> {
        let req = request.into_inner();
        debug!(
            "Received all_reduce for key={}, rank={}, op={:?}",
            req.key, req.rank, req.op
        );

        // For now, this is a simplified implementation
        // In production, this would use NCCL or implement ring-allreduce

        // Store the data temporarily (in reality, we'd aggregate from all ranks)
        let result_data = req.data.clone();

        Ok(Response::new(AllReduceResponse {
            success: true,
            message: "All-reduce completed".to_string(),
            data: result_data,
        }))
    }

    async fn broadcast(
        &self,
        request: Request<BroadcastRequest>,
    ) -> Result<Response<BroadcastResponse>, Status> {
        let req = request.into_inner();
        debug!(
            "Received broadcast for key={}, root={}, rank={}",
            req.key, req.root, req.rank
        );

        // Return the data as-is (simplified - would need proper coordination)
        Ok(Response::new(BroadcastResponse {
            success: true,
            message: "Broadcast completed".to_string(),
            data: req.data,
        }))
    }

    async fn gather(
        &self,
        request: Request<GatherRequest>,
    ) -> Result<Response<GatherResponse>, Status> {
        let req = request.into_inner();
        debug!(
            "Received gather for key={}, root={}, rank={}",
            req.key, req.root, req.rank
        );

        // Simplified gather - would need to collect from all ranks
        let gathered = vec![GatheredData {
            rank: req.rank,
            data: req.data,
        }];

        Ok(Response::new(GatherResponse {
            success: true,
            message: "Gather completed".to_string(),
            gathered,
        }))
    }

    async fn register_node(
        &self,
        request: Request<RegisterNodeRequest>,
    ) -> Result<Response<RegisterNodeResponse>, Status> {
        let req = request.into_inner();
        info!(
            "Registering node: rank={}, hostname={}, ip={}",
            req.rank, req.hostname, req.ip_address
        );

        let mut nodes = self.nodes.write().await;
        let mut world_size = self.world_size.write().await;

        let node_info = InternalNodeInfo {
            rank: req.rank,
            hostname: req.hostname.clone(),
            ip_address: req.ip_address.clone(),
            status: 1, // INITIALIZING
            last_heartbeat: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            capabilities: req.capabilities,
            resources: None,
        };

        nodes.insert(req.rank, node_info);
        *world_size = nodes.len() as u32;

        let node_addresses: Vec<String> = nodes
            .values()
            .map(|n| format!("{}:{}", n.ip_address, 50051))
            .collect();

        let cluster_config = Some(ClusterConfig {
            world_size: *world_size,
            node_addresses,
            master_rank: self.master_rank,
            master_address: "master:50051".to_string(),
        });

        Ok(Response::new(RegisterNodeResponse {
            success: true,
            message: "Node registered successfully".to_string(),
            assigned_rank: req.rank,
            world_size: *world_size,
            cluster_config,
        }))
    }

    async fn get_cluster_status(
        &self,
        request: Request<GetClusterStatusRequest>,
    ) -> Result<Response<GetClusterStatusResponse>, Status> {
        let req = request.into_inner();
        debug!("Received cluster status request from rank={}", req.rank);

        let nodes = self.nodes.read().await;
        let world_size = *self.world_size.read().await;

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let node_infos: Vec<NodeInfo> = nodes
            .values()
            .map(|n| {
                let status = Some(NodeStatus {
                    status: n.status,
                    message: String::new(),
                    last_update: n.last_heartbeat,
                });

                NodeInfo {
                    rank: n.rank,
                    hostname: n.hostname.clone(),
                    ip_address: n.ip_address.clone(),
                    status,
                    capabilities: n.capabilities.clone(),
                    resources: n.resources.clone(),
                }
            })
            .collect();

        let num_healthy = nodes
            .values()
            .filter(|n| now - n.last_heartbeat < 30000)
            .count();

        let num_failed = nodes.len() - num_healthy;

        let health = if num_failed == 0 {
            1 // HEALTHY
        } else if num_failed < nodes.len() / 2 {
            2 // DEGRADED
        } else {
            3 // UNHEALTHY
        };

        let cluster_health = Some(ClusterHealth {
            health,
            num_healthy_nodes: num_healthy as u32,
            num_failed_nodes: num_failed as u32,
            message: format!("{}/{} nodes healthy", num_healthy, nodes.len()),
        });

        Ok(Response::new(GetClusterStatusResponse {
            success: true,
            message: "Cluster status retrieved".to_string(),
            world_size,
            nodes: node_infos,
            health: cluster_health,
        }))
    }
}
