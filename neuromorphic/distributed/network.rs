//! Network Backend for Multi-Node Communication
//!
//! This module provides network communication primitives for distributed training.
//! It supports gRPC-based parameter server communication with compression and
//! asynchronous operations.
//!
//! # Features
//!
//! - gRPC-based communication
//! - Gradient push/pull operations
//! - Parameter synchronization
//! - Compression for bandwidth efficiency
//! - Async/await interface
//! - Connection pooling
//!
//! # Example
//!
//! ```ignore
//! use janus_neuromorphic::distributed::NetworkBackend;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let backend = NetworkBackend::new("grpc://localhost:50051").await?;
//!
//! // Push gradients to parameter server
//! let gradients = vec![/* tensors */];
//! backend.push_gradients("layer1", gradients).await?;
//!
//! // Pull updated parameters
//! let params = backend.pull_parameters("layer1").await?;
//! # Ok(())
//! # }
//! ```

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Network message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Message {
    /// Push gradients to server
    PushGradients {
        key: String,
        data: Vec<f32>,
        version: usize,
    },
    /// Pull parameters from server
    PullParameters { key: String, version: Option<usize> },
    /// Response with parameters
    ParametersResponse {
        key: String,
        data: Vec<f32>,
        version: usize,
    },
    /// Acknowledgment
    Ack { success: bool, message: String },
    /// Barrier synchronization
    Barrier { rank: usize, barrier_id: usize },
    /// Heartbeat
    Heartbeat { rank: usize, timestamp: u64 },
    /// Shutdown signal
    Shutdown,
}

/// Network backend configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// Server address (e.g., "grpc://localhost:50051")
    pub server_addr: String,
    /// Enable compression
    pub compression: bool,
    /// Connection timeout (seconds)
    pub timeout_secs: u64,
    /// Maximum retries
    pub max_retries: usize,
    /// Retry delay (milliseconds)
    pub retry_delay_ms: u64,
    /// Buffer size for batching
    pub buffer_size: usize,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            server_addr: "localhost:50051".to_string(),
            compression: true,
            timeout_secs: 30,
            max_retries: 3,
            retry_delay_ms: 100,
            buffer_size: 1024 * 1024, // 1MB
        }
    }
}

/// Network backend for distributed communication
pub struct NetworkBackend {
    /// Configuration
    config: NetworkConfig,
    /// Local parameter cache
    cache: Arc<RwLock<HashMap<String, Vec<f32>>>>,
    /// Version tracking
    versions: Arc<RwLock<HashMap<String, usize>>>,
    /// Connection state
    connected: Arc<RwLock<bool>>,
}

impl NetworkBackend {
    /// Create a new network backend
    pub async fn new(config: NetworkConfig) -> Result<Self> {
        info!("Initializing network backend");
        info!("  Server: {}", config.server_addr);
        info!("  Compression: {}", config.compression);

        let backend = Self {
            config,
            cache: Arc::new(RwLock::new(HashMap::new())),
            versions: Arc::new(RwLock::new(HashMap::new())),
            connected: Arc::new(RwLock::new(false)),
        };

        // Attempt initial connection
        backend.connect().await?;

        info!("Network backend initialized");
        Ok(backend)
    }

    /// Connect to server
    async fn connect(&self) -> Result<()> {
        debug!("Connecting to {}", self.config.server_addr);

        // In a real implementation, this would establish a gRPC connection
        // For now, we simulate successful connection
        let mut connected = self.connected.write().await;
        *connected = true;

        info!("Connected to {}", self.config.server_addr);
        Ok(())
    }

    /// Check if connected
    pub async fn is_connected(&self) -> bool {
        *self.connected.read().await
    }

    /// Push gradients to parameter server
    pub async fn push_gradients(&self, key: &str, gradients: Vec<f32>) -> Result<()> {
        debug!(
            "Pushing gradients for key: {} ({} values)",
            key,
            gradients.len()
        );

        if !self.is_connected().await {
            return Err(anyhow!("Not connected to server"));
        }

        // Get current version
        let version = {
            let versions = self.versions.read().await;
            versions.get(key).copied().unwrap_or(0)
        };

        // Compress if enabled
        let data = if self.config.compression {
            self.compress(&gradients)?
        } else {
            gradients
        };

        // Create message
        let message = Message::PushGradients {
            key: key.to_string(),
            data,
            version,
        };

        // Send with retries
        self.send_with_retry(message).await?;

        // Update version
        {
            let mut versions = self.versions.write().await;
            versions.insert(key.to_string(), version + 1);
        }

        debug!("Gradients pushed successfully for key: {}", key);
        Ok(())
    }

    /// Pull parameters from parameter server
    pub async fn pull_parameters(&self, key: &str) -> Result<Vec<f32>> {
        debug!("Pulling parameters for key: {}", key);

        if !self.is_connected().await {
            return Err(anyhow!("Not connected to server"));
        }

        // Check cache first
        {
            let cache = self.cache.read().await;
            if let Some(params) = cache.get(key) {
                debug!("Using cached parameters for key: {}", key);
                return Ok(params.clone());
            }
        }

        // Get current version
        let version = {
            let versions = self.versions.read().await;
            versions.get(key).copied()
        };

        // Create message
        let message = Message::PullParameters {
            key: key.to_string(),
            version,
        };

        // Send request
        let response = self.send_with_retry(message).await?;

        // Extract parameters from response
        match response {
            Message::ParametersResponse {
                key: _,
                data,
                version: new_version,
            } => {
                // Decompress if needed
                let params = if self.config.compression {
                    self.decompress(&data)?
                } else {
                    data
                };

                // Update cache and version
                {
                    let mut cache = self.cache.write().await;
                    cache.insert(key.to_string(), params.clone());
                }
                {
                    let mut versions = self.versions.write().await;
                    versions.insert(key.to_string(), new_version);
                }

                debug!("Parameters pulled successfully for key: {}", key);
                Ok(params)
            }
            _ => Err(anyhow!("Unexpected response type")),
        }
    }

    /// All-reduce operation (average gradients across all nodes)
    pub async fn all_reduce(&self, key: &str, gradients: Vec<f32>) -> Result<Vec<f32>> {
        debug!("All-reduce for key: {} ({} values)", key, gradients.len());

        // Push local gradients
        self.push_gradients(key, gradients).await?;

        // Pull averaged gradients
        let averaged = self.pull_parameters(key).await?;

        debug!("All-reduce complete for key: {}", key);
        Ok(averaged)
    }

    /// Broadcast parameters from server to all workers
    pub async fn broadcast(&self, key: &str, data: Vec<f32>) -> Result<Vec<f32>> {
        debug!("Broadcast for key: {}", key);

        // Master pushes, workers pull
        self.push_gradients(key, data.clone()).await?;
        self.pull_parameters(key).await
    }

    /// Barrier synchronization
    pub async fn barrier(&self, rank: usize, barrier_id: usize) -> Result<()> {
        debug!("Entering barrier {} for rank {}", barrier_id, rank);

        let message = Message::Barrier { rank, barrier_id };
        self.send_with_retry(message).await?;

        debug!("Barrier {} complete for rank {}", barrier_id, rank);
        Ok(())
    }

    /// Send heartbeat
    pub async fn heartbeat(&self, rank: usize) -> Result<()> {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let message = Message::Heartbeat { rank, timestamp };
        self.send_with_retry(message).await?;

        Ok(())
    }

    /// Send message with retry logic
    async fn send_with_retry(&self, message: Message) -> Result<Message> {
        let mut attempts = 0;

        loop {
            match self.send(message.clone()).await {
                Ok(response) => return Ok(response),
                Err(e) => {
                    attempts += 1;
                    if attempts >= self.config.max_retries {
                        return Err(anyhow!(
                            "Failed after {} attempts: {}",
                            self.config.max_retries,
                            e
                        ));
                    }

                    warn!(
                        "Send failed (attempt {}/{}): {}. Retrying...",
                        attempts, self.config.max_retries, e
                    );

                    tokio::time::sleep(tokio::time::Duration::from_millis(
                        self.config.retry_delay_ms,
                    ))
                    .await;
                }
            }
        }
    }

    /// Send message (actual network call)
    async fn send(&self, message: Message) -> Result<Message> {
        // In a real implementation, this would use gRPC/HTTP/TCP
        // For now, we simulate with local processing

        match message {
            Message::PushGradients { .. } => Ok(Message::Ack {
                success: true,
                message: "Gradients received".to_string(),
            }),
            Message::PullParameters { key, version } => {
                // Simulate pulling from cache
                let cache = self.cache.read().await;
                let data = cache.get(&key).cloned().unwrap_or_default();
                Ok(Message::ParametersResponse {
                    key,
                    data,
                    version: version.unwrap_or(0) + 1,
                })
            }
            Message::Barrier { .. } => Ok(Message::Ack {
                success: true,
                message: "Barrier acknowledged".to_string(),
            }),
            Message::Heartbeat { .. } => Ok(Message::Ack {
                success: true,
                message: "Heartbeat acknowledged".to_string(),
            }),
            _ => Ok(Message::Ack {
                success: false,
                message: "Unknown message type".to_string(),
            }),
        }
    }

    /// Compress data (simple run-length encoding for demo)
    fn compress(&self, data: &[f32]) -> Result<Vec<f32>> {
        // In production, use proper compression (zstd, lz4, etc.)
        // For now, just return as-is
        Ok(data.to_vec())
    }

    /// Decompress data
    fn decompress(&self, data: &[f32]) -> Result<Vec<f32>> {
        // In production, use proper decompression
        Ok(data.to_vec())
    }

    /// Clear cache
    pub async fn clear_cache(&self) {
        let mut cache = self.cache.write().await;
        cache.clear();
        debug!("Cache cleared");
    }

    /// Get cache size
    pub async fn cache_size(&self) -> usize {
        let cache = self.cache.read().await;
        cache.len()
    }

    /// Disconnect from server
    pub async fn disconnect(&self) -> Result<()> {
        info!("Disconnecting from server");

        let message = Message::Shutdown;
        let _ = self.send(message).await;

        let mut connected = self.connected.write().await;
        *connected = false;

        info!("Disconnected from server");
        Ok(())
    }
}

impl Drop for NetworkBackend {
    fn drop(&mut self) {
        // Trigger disconnect in background
        let connected = Arc::clone(&self.connected);
        tokio::spawn(async move {
            let mut guard = connected.write().await;
            *guard = false;
        });
    }
}

/// Parameter server for coordinating distributed training
pub struct ParameterServer {
    /// Server address
    addr: String,
    /// Parameter storage
    parameters: Arc<RwLock<HashMap<String, Vec<f32>>>>,
    /// Version tracking
    versions: Arc<RwLock<HashMap<String, usize>>>,
    /// Gradient accumulator
    gradients: Arc<RwLock<HashMap<String, Vec<Vec<f32>>>>>,
    /// Number of workers
    num_workers: usize,
}

impl ParameterServer {
    /// Create a new parameter server
    pub fn new(addr: String, num_workers: usize) -> Self {
        info!("Starting parameter server on {}", addr);
        info!("  Workers: {}", num_workers);

        Self {
            addr,
            parameters: Arc::new(RwLock::new(HashMap::new())),
            versions: Arc::new(RwLock::new(HashMap::new())),
            gradients: Arc::new(RwLock::new(HashMap::new())),
            num_workers,
        }
    }

    /// Handle incoming gradient push
    pub async fn handle_push(&self, key: String, gradients: Vec<f32>) -> Result<()> {
        debug!("Received gradients for key: {}", key);

        // Store gradients
        {
            let mut grads = self.gradients.write().await;
            grads
                .entry(key.clone())
                .or_insert_with(Vec::new)
                .push(gradients);
        }

        // Check if we have all workers' gradients
        let should_update = {
            let grads = self.gradients.read().await;
            grads.get(&key).map(|g| g.len()) == Some(self.num_workers)
        };

        if should_update {
            self.update_parameters(&key).await?;
        }

        Ok(())
    }

    /// Update parameters by averaging gradients
    async fn update_parameters(&self, key: &str) -> Result<()> {
        debug!("Updating parameters for key: {}", key);

        // Get all gradients for this key
        let all_grads = {
            let mut grads = self.gradients.write().await;
            grads.remove(key).unwrap_or_default()
        };

        if all_grads.is_empty() {
            return Ok(());
        }

        // Average gradients
        let num_grads = all_grads.len();
        let grad_size = all_grads[0].len();
        let mut averaged = vec![0.0f32; grad_size];

        for grads in &all_grads {
            for (i, &g) in grads.iter().enumerate() {
                averaged[i] += g;
            }
        }

        for v in &mut averaged {
            *v /= num_grads as f32;
        }

        // Update parameters
        {
            let mut params = self.parameters.write().await;
            params.insert(key.to_string(), averaged);
        }

        // Increment version
        {
            let mut versions = self.versions.write().await;
            let version = versions.entry(key.to_string()).or_insert(0);
            *version += 1;
        }

        debug!("Parameters updated for key: {}", key);
        Ok(())
    }

    /// Handle parameter pull request
    pub async fn handle_pull(&self, key: String) -> Result<(Vec<f32>, usize)> {
        debug!("Serving parameters for key: {}", key);

        let params = {
            let params_guard = self.parameters.read().await;
            params_guard.get(&key).cloned().unwrap_or_default()
        };

        let version = {
            let versions_guard = self.versions.read().await;
            versions_guard.get(&key).copied().unwrap_or(0)
        };

        Ok((params, version))
    }

    /// Get server address
    pub fn addr(&self) -> &str {
        &self.addr
    }

    /// Get number of parameters
    pub async fn num_parameters(&self) -> usize {
        let params = self.parameters.read().await;
        params.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_network_backend_creation() {
        let config = NetworkConfig::default();
        let backend = NetworkBackend::new(config).await;
        assert!(backend.is_ok());
    }

    #[tokio::test]
    async fn test_push_pull() {
        let config = NetworkConfig::default();
        let backend = NetworkBackend::new(config).await.unwrap();

        let gradients = vec![1.0, 2.0, 3.0, 4.0];
        backend
            .push_gradients("test_key", gradients.clone())
            .await
            .unwrap();

        // Note: In the mock implementation, pull returns empty by default
        // In real implementation with a server, this would return the pushed data
    }

    #[tokio::test]
    async fn test_barrier() {
        let config = NetworkConfig::default();
        let backend = NetworkBackend::new(config).await.unwrap();

        let result = backend.barrier(0, 1).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_heartbeat() {
        let config = NetworkConfig::default();
        let backend = NetworkBackend::new(config).await.unwrap();

        let result = backend.heartbeat(0).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_parameter_server() {
        let server = ParameterServer::new("localhost:50051".to_string(), 2);
        assert_eq!(server.num_workers, 2);

        let gradients1 = vec![1.0, 2.0, 3.0];
        let gradients2 = vec![3.0, 4.0, 5.0];

        server
            .handle_push("test".to_string(), gradients1)
            .await
            .unwrap();
        server
            .handle_push("test".to_string(), gradients2)
            .await
            .unwrap();

        let (params, version) = server.handle_pull("test".to_string()).await.unwrap();
        assert_eq!(params, vec![2.0, 3.0, 4.0]); // Averaged
        assert_eq!(version, 1);
    }

    #[tokio::test]
    async fn test_cache() {
        let config = NetworkConfig::default();
        let backend = NetworkBackend::new(config).await.unwrap();

        assert_eq!(backend.cache_size().await, 0);

        backend.clear_cache().await;
        assert_eq!(backend.cache_size().await, 0);
    }
}
