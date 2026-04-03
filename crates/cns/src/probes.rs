//! # Health Probes Module
//!
//! Implements health check probes for all JANUS system components.
//! These probes act as the "sensory neurons" of the CNS, detecting the state of each component.

use crate::Result;
use crate::signals::{ComponentHealth, ComponentType, ProbeStatus};
use anyhow::Context;
use async_trait::async_trait;
use chrono::Utc;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::time::{Duration, Instant};
use tokio::time::timeout;
use tonic::transport::Channel;
use tracing::{debug, warn};

/// Default timeout for health probes
const DEFAULT_PROBE_TIMEOUT: Duration = Duration::from_secs(5);

/// Result of a health probe execution
#[derive(Debug, Clone)]
pub struct ProbeResult {
    /// Component that was probed
    pub component_type: ComponentType,
    /// Probe status
    pub status: ProbeStatus,
    /// Status message
    pub message: String,
    /// Response time in milliseconds
    pub response_time_ms: u64,
}

impl ProbeResult {
    /// Convert to ComponentHealth
    pub fn to_component_health(self) -> ComponentHealth {
        ComponentHealth {
            component_type: self.component_type,
            status: self.status,
            message: self.message,
            last_check: Utc::now(),
            response_time_ms: Some(self.response_time_ms),
            metadata: Default::default(),
        }
    }
}

/// Health probe trait - all probes must implement this
#[async_trait]
pub trait HealthProbe: Send + Sync {
    /// The component type this probe checks
    fn component_type(&self) -> ComponentType;

    /// Execute the health check
    async fn check(&self) -> Result<ProbeResult>;

    /// Get the probe timeout
    fn timeout(&self) -> Duration {
        DEFAULT_PROBE_TIMEOUT
    }

    /// Execute the probe with timeout
    async fn check_with_timeout(&self) -> ProbeResult {
        let start = Instant::now();
        let component_type = self.component_type();

        match timeout(self.timeout(), self.check()).await {
            Ok(Ok(result)) => result,
            Ok(Err(e)) => {
                warn!(
                    component = %component_type,
                    error = %e,
                    "Health probe failed"
                );
                ProbeResult {
                    component_type,
                    status: ProbeStatus::Down,
                    message: format!("Probe error: {}", e),
                    response_time_ms: start.elapsed().as_millis() as u64,
                }
            }
            Err(_) => {
                warn!(
                    component = %component_type,
                    timeout_ms = ?self.timeout().as_millis(),
                    "Health probe timed out"
                );
                ProbeResult {
                    component_type,
                    status: ProbeStatus::Down,
                    message: "Probe timeout".to_string(),
                    response_time_ms: start.elapsed().as_millis() as u64,
                }
            }
        }
    }
}

// ============================================================================
// Service Probes
// ============================================================================

/// HTTP-based health probe for services with HTTP endpoints
pub struct HttpHealthProbe {
    component_type: ComponentType,
    url: String,
    client: reqwest::Client,
}

impl HttpHealthProbe {
    /// Create a new HTTP health probe
    pub fn new(component_type: ComponentType, url: impl Into<String>) -> Self {
        Self {
            component_type,
            url: url.into(),
            client: reqwest::Client::builder()
                .timeout(DEFAULT_PROBE_TIMEOUT)
                .build()
                .unwrap(),
        }
    }
}

#[async_trait]
impl HealthProbe for HttpHealthProbe {
    fn component_type(&self) -> ComponentType {
        self.component_type
    }

    async fn check(&self) -> Result<ProbeResult> {
        let start = Instant::now();

        let response = self
            .client
            .get(&self.url)
            .send()
            .await
            .context("HTTP request failed")?;

        let status = if response.status().is_success() {
            ProbeStatus::Up
        } else if response.status().is_server_error() {
            ProbeStatus::Down
        } else {
            ProbeStatus::Degraded
        };

        Ok(ProbeResult {
            component_type: self.component_type,
            status,
            message: format!("HTTP {}", response.status()),
            response_time_ms: start.elapsed().as_millis() as u64,
        })
    }
}

/// gRPC health probe using standard health check protocol
pub struct GrpcHealthProbe {
    component_type: ComponentType,
    endpoint: String,
    service_name: Option<String>,
}

impl GrpcHealthProbe {
    /// Create a new gRPC health probe
    pub fn new(component_type: ComponentType, endpoint: impl Into<String>) -> Self {
        Self {
            component_type,
            endpoint: endpoint.into(),
            service_name: None,
        }
    }

    /// Create a new gRPC health probe with a specific service name
    pub fn with_service(
        component_type: ComponentType,
        endpoint: impl Into<String>,
        service_name: impl Into<String>,
    ) -> Self {
        Self {
            component_type,
            endpoint: endpoint.into(),
            service_name: Some(service_name.into()),
        }
    }
}

#[async_trait]
impl HealthProbe for GrpcHealthProbe {
    fn component_type(&self) -> ComponentType {
        self.component_type
    }

    async fn check(&self) -> Result<ProbeResult> {
        let start = Instant::now();

        match self.check_health_protocol().await {
            Ok(serving) => {
                let status = if serving {
                    ProbeStatus::Up
                } else {
                    ProbeStatus::Degraded
                };

                Ok(ProbeResult {
                    component_type: self.component_type,
                    status,
                    message: format!(
                        "gRPC health check: {}",
                        if serving { "SERVING" } else { "NOT_SERVING" }
                    ),
                    response_time_ms: start.elapsed().as_millis() as u64,
                })
            }
            Err(e) => {
                warn!("gRPC health check failed for {}: {}", self.endpoint, e);
                Ok(ProbeResult {
                    component_type: self.component_type,
                    status: ProbeStatus::Down,
                    message: format!("gRPC health check failed: {}", e),
                    response_time_ms: start.elapsed().as_millis() as u64,
                })
            }
        }
    }
}

impl GrpcHealthProbe {
    /// Perform gRPC health check using the standard health checking protocol
    async fn check_health_protocol(&self) -> Result<bool> {
        use tonic_health::pb::HealthCheckRequest;
        use tonic_health::pb::health_client::HealthClient;

        // Connect to the gRPC service
        let channel = Channel::from_shared(self.endpoint.clone())
            .context("Invalid gRPC endpoint")?
            .connect_timeout(Duration::from_secs(3))
            .timeout(Duration::from_secs(5))
            .connect()
            .await
            .context("Failed to connect to gRPC service")?;

        let mut client = HealthClient::new(channel);

        // Prepare health check request
        let request = HealthCheckRequest {
            service: self.service_name.clone().unwrap_or_default(),
        };

        debug!(
            "Checking gRPC health for service: {}",
            self.service_name.as_deref().unwrap_or("(all)")
        );

        // Perform health check
        let response = client
            .check(request)
            .await
            .context("Health check RPC failed")?;

        // Parse response status
        use tonic_health::pb::health_check_response::ServingStatus;
        let serving_status = response.into_inner().status;

        Ok(serving_status == ServingStatus::Serving as i32)
    }

    /// Legacy TCP connection check (fallback)
    #[allow(dead_code)]
    async fn can_connect(&self) -> bool {
        tokio::net::TcpStream::connect(&self.endpoint).await.is_ok()
    }
}

// ============================================================================
// Dependency Probes
// ============================================================================

/// Redis health probe
pub struct RedisHealthProbe {
    client: redis::Client,
}

impl RedisHealthProbe {
    /// Create a new Redis health probe
    ///
    /// Returns an error if the connection string is invalid.
    /// The client is reused across health checks to avoid connection churn.
    pub fn new(connection_string: impl Into<String>) -> Result<Self> {
        let connection_string = connection_string.into();
        let client = redis::Client::open(connection_string.as_str())
            .context("Failed to create Redis client")?;
        Ok(Self { client })
    }
}

#[async_trait]
impl HealthProbe for RedisHealthProbe {
    fn component_type(&self) -> ComponentType {
        ComponentType::Redis
    }

    async fn check(&self) -> Result<ProbeResult> {
        let start = Instant::now();

        // Reuse the client to get a connection from the pool
        let mut con = self
            .client
            .get_multiplexed_async_connection()
            .await
            .context("Failed to connect to Redis")?;

        // Execute PING command
        let pong: String = redis::cmd("PING")
            .query_async(&mut con)
            .await
            .context("Redis PING failed")?;

        let status = if pong == "PONG" {
            ProbeStatus::Up
        } else {
            ProbeStatus::Degraded
        };

        Ok(ProbeResult {
            component_type: ComponentType::Redis,
            status,
            message: format!("Redis responded: {}", pong),
            response_time_ms: start.elapsed().as_millis() as u64,
        })
    }
}

/// Qdrant vector database health probe
pub struct QdrantHealthProbe {
    url: String,
    client: reqwest::Client,
}

impl QdrantHealthProbe {
    /// Create a new Qdrant health probe
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            client: reqwest::Client::builder()
                .timeout(DEFAULT_PROBE_TIMEOUT)
                .build()
                .unwrap(),
        }
    }
}

#[async_trait]
impl HealthProbe for QdrantHealthProbe {
    fn component_type(&self) -> ComponentType {
        ComponentType::Qdrant
    }

    async fn check(&self) -> Result<ProbeResult> {
        let start = Instant::now();

        // Qdrant health endpoint
        let health_url = format!("{}/healthz", self.url);

        let response = self
            .client
            .get(&health_url)
            .send()
            .await
            .context("Failed to connect to Qdrant")?;

        let status = if response.status().is_success() {
            ProbeStatus::Up
        } else {
            ProbeStatus::Down
        };

        Ok(ProbeResult {
            component_type: ComponentType::Qdrant,
            status,
            message: format!("Qdrant health: {}", response.status()),
            response_time_ms: start.elapsed().as_millis() as u64,
        })
    }
}

// ============================================================================
// Communication Channel Probes
// ============================================================================

/// Shared memory health probe
pub struct SharedMemoryProbe {
    shm_path: String,
    min_size_bytes: Option<u64>,
    expected_permissions: Option<u32>,
}

impl SharedMemoryProbe {
    /// Create a new shared memory probe
    pub fn new(shm_path: impl Into<String>) -> Self {
        Self {
            shm_path: shm_path.into(),
            min_size_bytes: None,
            expected_permissions: None,
        }
    }

    /// Create a probe with size validation
    pub fn with_min_size(mut self, min_size_bytes: u64) -> Self {
        self.min_size_bytes = Some(min_size_bytes);
        self
    }

    /// Create a probe with permission validation (Unix mode)
    pub fn with_permissions(mut self, permissions: u32) -> Self {
        self.expected_permissions = Some(permissions);
        self
    }
}

#[async_trait]
impl HealthProbe for SharedMemoryProbe {
    fn component_type(&self) -> ComponentType {
        ComponentType::SharedMemory
    }

    async fn check(&self) -> Result<ProbeResult> {
        let start = Instant::now();

        // Perform comprehensive shared memory validation
        match self.validate_shared_memory() {
            Ok(message) => Ok(ProbeResult {
                component_type: ComponentType::SharedMemory,
                status: ProbeStatus::Up,
                message,
                response_time_ms: start.elapsed().as_millis() as u64,
            }),
            Err(e) => {
                warn!("Shared memory check failed for {}: {}", self.shm_path, e);
                Ok(ProbeResult {
                    component_type: ComponentType::SharedMemory,
                    status: ProbeStatus::Down,
                    message: format!("Shared memory validation failed: {}", e),
                    response_time_ms: start.elapsed().as_millis() as u64,
                })
            }
        }
    }
}

impl SharedMemoryProbe {
    /// Comprehensive shared memory validation
    fn validate_shared_memory(&self) -> Result<String> {
        let path = std::path::Path::new(&self.shm_path);

        // 1. Check existence
        if !path.exists() {
            return Err(crate::CNSError::ProbeFailure(format!(
                "Shared memory segment does not exist: {}",
                self.shm_path
            )));
        }

        // 2. Check if it's a file (shared memory segments appear as files in /dev/shm)
        let metadata = fs::metadata(path).map_err(|e| {
            crate::CNSError::ProbeFailure(format!("Failed to read shared memory metadata: {}", e))
        })?;

        if !metadata.is_file() {
            return Err(crate::CNSError::ProbeFailure(format!(
                "Shared memory path is not a file: {}",
                self.shm_path
            )));
        }

        // 3. Validate size if specified
        let size = metadata.len();
        if let Some(min_size) = self.min_size_bytes
            && size < min_size
        {
            return Err(crate::CNSError::ProbeFailure(format!(
                "Shared memory size {} bytes is below minimum {} bytes",
                size, min_size
            )));
        }

        // 4. Validate permissions if specified (Unix only)
        #[cfg(unix)]
        if let Some(expected_perms) = self.expected_permissions {
            let actual_perms = metadata.permissions().mode() & 0o777;
            if actual_perms != expected_perms {
                return Err(crate::CNSError::ProbeFailure(format!(
                    "Shared memory permissions {:o} do not match expected {:o}",
                    actual_perms, expected_perms
                )));
            }
        }

        // 5. Check read/write accessibility
        if metadata.permissions().readonly() {
            return Err(crate::CNSError::ProbeFailure(
                "Shared memory is read-only".to_string(),
            ));
        }

        // 6. Verify we can open the file
        fs::File::open(path).map_err(|e| {
            crate::CNSError::ProbeFailure(format!("Failed to open shared memory segment: {}", e))
        })?;

        Ok(format!(
            "Shared memory accessible (size: {} bytes, path: {})",
            size, self.shm_path
        ))
    }

    /// Legacy simple accessibility check
    #[allow(dead_code)]
    fn check_shm_accessible(&self) -> bool {
        std::path::Path::new(&self.shm_path).exists()
    }
}

// ============================================================================
// Composite Probe
// ============================================================================

/// Probe that aggregates multiple probes for a component
pub struct CompositeProbe {
    component_type: ComponentType,
    probes: Vec<Box<dyn HealthProbe>>,
}

impl CompositeProbe {
    /// Create a new composite probe
    pub fn new(component_type: ComponentType) -> Self {
        Self {
            component_type,
            probes: Vec::new(),
        }
    }

    /// Add a probe to the composite
    pub fn add_probe(mut self, probe: Box<dyn HealthProbe>) -> Self {
        self.probes.push(probe);
        self
    }
}

#[async_trait]
impl HealthProbe for CompositeProbe {
    fn component_type(&self) -> ComponentType {
        self.component_type
    }

    async fn check(&self) -> Result<ProbeResult> {
        let start = Instant::now();

        if self.probes.is_empty() {
            return Ok(ProbeResult {
                component_type: self.component_type,
                status: ProbeStatus::Unknown,
                message: "No probes configured".to_string(),
                response_time_ms: 0,
            });
        }

        // Execute all probes concurrently
        let mut results = Vec::new();
        for probe in &self.probes {
            results.push(probe.check_with_timeout().await);
        }

        // Aggregate results
        let all_up = results.iter().all(|r| r.status == ProbeStatus::Up);
        let any_down = results.iter().any(|r| r.status == ProbeStatus::Down);

        let status = if all_up {
            ProbeStatus::Up
        } else if any_down {
            ProbeStatus::Down
        } else {
            ProbeStatus::Degraded
        };

        let messages: Vec<String> = results.iter().map(|r| r.message.clone()).collect();

        Ok(ProbeResult {
            component_type: self.component_type,
            status,
            message: messages.join("; "),
            response_time_ms: start.elapsed().as_millis() as u64,
        })
    }
}

// ============================================================================
// Probe Builder
// ============================================================================

/// Builder for creating standard JANUS probes
pub struct ProbeBuilder;

impl ProbeBuilder {
    /// Create probe for Forward service
    pub fn forward_service(base_url: &str) -> Box<dyn HealthProbe> {
        Box::new(HttpHealthProbe::new(
            ComponentType::ForwardService,
            format!("{}/health", base_url),
        ))
    }

    /// Create probe for Backward service
    pub fn backward_service(base_url: &str) -> Box<dyn HealthProbe> {
        Box::new(HttpHealthProbe::new(
            ComponentType::BackwardService,
            format!("{}/health", base_url),
        ))
    }

    /// Create probe for Gateway service
    pub fn gateway_service(base_url: &str) -> Box<dyn HealthProbe> {
        Box::new(HttpHealthProbe::new(
            ComponentType::GatewayService,
            format!("{}/health", base_url),
        ))
    }

    /// Create probe for Redis
    pub fn redis(connection_string: &str) -> Box<dyn HealthProbe> {
        Box::new(
            RedisHealthProbe::new(connection_string).expect("Failed to create Redis health probe"),
        )
    }

    /// Create probe for Qdrant
    pub fn qdrant(url: &str) -> Box<dyn HealthProbe> {
        Box::new(QdrantHealthProbe::new(url))
    }

    /// Create probe for shared memory
    pub fn shared_memory(path: &str) -> Box<dyn HealthProbe> {
        Box::new(SharedMemoryProbe::new(path))
    }

    /// Create probe for gRPC service
    pub fn grpc_service(component_type: ComponentType, endpoint: &str) -> Box<dyn HealthProbe> {
        Box::new(GrpcHealthProbe::new(component_type, endpoint))
    }

    /// Create probe for neuromorphic brain region
    pub fn neuromorphic_region(component_type: ComponentType, url: &str) -> Box<dyn HealthProbe> {
        Box::new(HttpHealthProbe::new(
            component_type,
            format!("{}/health", url),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_probe_result_to_component_health() {
        let result = ProbeResult {
            component_type: ComponentType::ForwardService,
            status: ProbeStatus::Up,
            message: "OK".to_string(),
            response_time_ms: 42,
        };

        let health = result.to_component_health();
        assert_eq!(health.component_type, ComponentType::ForwardService);
        assert_eq!(health.status, ProbeStatus::Up);
        assert_eq!(health.response_time_ms, Some(42));
    }

    #[tokio::test]
    async fn test_composite_probe_all_up() {
        // Create a simple mock probe
        struct MockProbe(ProbeStatus);

        #[async_trait]
        impl HealthProbe for MockProbe {
            fn component_type(&self) -> ComponentType {
                ComponentType::ForwardService
            }

            async fn check(&self) -> Result<ProbeResult> {
                Ok(ProbeResult {
                    component_type: ComponentType::ForwardService,
                    status: self.0,
                    message: "Mock".to_string(),
                    response_time_ms: 1,
                })
            }
        }

        let composite = CompositeProbe::new(ComponentType::ForwardService)
            .add_probe(Box::new(MockProbe(ProbeStatus::Up)))
            .add_probe(Box::new(MockProbe(ProbeStatus::Up)));

        let result = composite.check().await.unwrap();
        assert_eq!(result.status, ProbeStatus::Up);
    }

    #[tokio::test]
    async fn test_composite_probe_degraded() {
        struct MockProbe(ProbeStatus);

        #[async_trait]
        impl HealthProbe for MockProbe {
            fn component_type(&self) -> ComponentType {
                ComponentType::ForwardService
            }

            async fn check(&self) -> Result<ProbeResult> {
                Ok(ProbeResult {
                    component_type: ComponentType::ForwardService,
                    status: self.0,
                    message: "Mock".to_string(),
                    response_time_ms: 1,
                })
            }
        }

        let composite = CompositeProbe::new(ComponentType::ForwardService)
            .add_probe(Box::new(MockProbe(ProbeStatus::Up)))
            .add_probe(Box::new(MockProbe(ProbeStatus::Degraded)));

        let result = composite.check().await.unwrap();
        assert_eq!(result.status, ProbeStatus::Degraded);
    }
}
