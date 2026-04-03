//! # Health Signals Module
//!
//! Defines the types and structures for health signals throughout the CNS system.
//! Analogous to neural signals in the biological nervous system.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// Overall system health status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SystemStatus {
    /// All systems operational
    Healthy,
    /// Some non-critical issues detected
    Degraded,
    /// Critical issues, immediate attention required
    Critical,
    /// System is shutting down
    Shutdown,
    /// System is starting up
    Starting,
}

impl fmt::Display for SystemStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SystemStatus::Healthy => write!(f, "HEALTHY"),
            SystemStatus::Degraded => write!(f, "DEGRADED"),
            SystemStatus::Critical => write!(f, "CRITICAL"),
            SystemStatus::Shutdown => write!(f, "SHUTDOWN"),
            SystemStatus::Starting => write!(f, "STARTING"),
        }
    }
}

impl SystemStatus {
    /// Returns true if the status indicates the system is operational
    pub fn is_operational(&self) -> bool {
        matches!(self, SystemStatus::Healthy | SystemStatus::Degraded)
    }

    /// Returns true if the status indicates critical failure
    pub fn is_critical(&self) -> bool {
        matches!(self, SystemStatus::Critical)
    }

    /// Aggregate status from multiple component statuses
    pub fn aggregate(statuses: &[ComponentHealth]) -> Self {
        if statuses.is_empty() {
            return SystemStatus::Starting;
        }

        let has_critical = statuses.iter().any(|h| h.status == ProbeStatus::Down);
        let has_degraded = statuses.iter().any(|h| h.status == ProbeStatus::Degraded);

        if has_critical {
            SystemStatus::Critical
        } else if has_degraded {
            SystemStatus::Degraded
        } else {
            SystemStatus::Healthy
        }
    }
}

/// Health status for individual probes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProbeStatus {
    /// Component is fully operational
    Up,
    /// Component is operational but degraded
    Degraded,
    /// Component is not operational
    Down,
    /// Component status is unknown
    Unknown,
}

impl fmt::Display for ProbeStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProbeStatus::Up => write!(f, "UP"),
            ProbeStatus::Degraded => write!(f, "DEGRADED"),
            ProbeStatus::Down => write!(f, "DOWN"),
            ProbeStatus::Unknown => write!(f, "UNKNOWN"),
        }
    }
}

impl ProbeStatus {
    /// Convert to a numeric health score (0.0 = down, 1.0 = up)
    pub fn score(&self) -> f64 {
        match self {
            ProbeStatus::Up => 1.0,
            ProbeStatus::Degraded => 0.5,
            ProbeStatus::Down => 0.0,
            ProbeStatus::Unknown => 0.25,
        }
    }
}

/// Component types in the JANUS system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComponentType {
    // Core Services
    ForwardService,
    BackwardService,
    GatewayService,
    CNSService,

    // Dependencies
    Redis,
    Qdrant,

    // Communication Channels
    SharedMemory,
    GrpcChannel,
    WebSocket,

    // Internal Subsystems
    VisionModule,
    LogicModule,
    MemoryModule,
    ExecutionModule,

    // Infrastructure
    JobQueue,
    MetricsExporter,

    // Neuromorphic Brain Regions
    Cortex,
    Hippocampus,
    BasalGanglia,
    Thalamus,
    Prefrontal,
    Amygdala,
    Hypothalamus,
    Cerebellum,
    VisualCortex,
    Integration,
}

impl fmt::Display for ComponentType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ComponentType::ForwardService => write!(f, "forward_service"),
            ComponentType::BackwardService => write!(f, "backward_service"),
            ComponentType::GatewayService => write!(f, "gateway_service"),
            ComponentType::CNSService => write!(f, "cns_service"),
            ComponentType::Redis => write!(f, "redis"),
            ComponentType::Qdrant => write!(f, "qdrant"),
            ComponentType::SharedMemory => write!(f, "shared_memory"),
            ComponentType::GrpcChannel => write!(f, "grpc_channel"),
            ComponentType::WebSocket => write!(f, "websocket"),
            ComponentType::VisionModule => write!(f, "vision_module"),
            ComponentType::LogicModule => write!(f, "logic_module"),
            ComponentType::MemoryModule => write!(f, "memory_module"),
            ComponentType::ExecutionModule => write!(f, "execution_module"),
            ComponentType::JobQueue => write!(f, "job_queue"),
            ComponentType::MetricsExporter => write!(f, "metrics_exporter"),
            ComponentType::Cortex => write!(f, "cortex"),
            ComponentType::Hippocampus => write!(f, "hippocampus"),
            ComponentType::BasalGanglia => write!(f, "basal_ganglia"),
            ComponentType::Thalamus => write!(f, "thalamus"),
            ComponentType::Prefrontal => write!(f, "prefrontal"),
            ComponentType::Amygdala => write!(f, "amygdala"),
            ComponentType::Hypothalamus => write!(f, "hypothalamus"),
            ComponentType::Cerebellum => write!(f, "cerebellum"),
            ComponentType::VisualCortex => write!(f, "visual_cortex"),
            ComponentType::Integration => write!(f, "integration"),
        }
    }
}

/// Health information for a specific component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    /// Type of component
    pub component_type: ComponentType,
    /// Current status
    pub status: ProbeStatus,
    /// Human-readable message
    pub message: String,
    /// Last check timestamp
    pub last_check: DateTime<Utc>,
    /// Response time in milliseconds
    pub response_time_ms: Option<u64>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl ComponentHealth {
    /// Create a new healthy component status
    pub fn healthy(component_type: ComponentType) -> Self {
        Self {
            component_type,
            status: ProbeStatus::Up,
            message: "OK".to_string(),
            last_check: Utc::now(),
            response_time_ms: None,
            metadata: HashMap::new(),
        }
    }

    /// Create a degraded component status
    pub fn degraded(component_type: ComponentType, message: impl Into<String>) -> Self {
        Self {
            component_type,
            status: ProbeStatus::Degraded,
            message: message.into(),
            last_check: Utc::now(),
            response_time_ms: None,
            metadata: HashMap::new(),
        }
    }

    /// Create an unhealthy component status
    pub fn unhealthy(component_type: ComponentType, message: impl Into<String>) -> Self {
        Self {
            component_type,
            status: ProbeStatus::Down,
            message: message.into(),
            last_check: Utc::now(),
            response_time_ms: None,
            metadata: HashMap::new(),
        }
    }

    /// Create an unknown component status
    pub fn unknown(component_type: ComponentType) -> Self {
        Self {
            component_type,
            status: ProbeStatus::Unknown,
            message: "Status unknown".to_string(),
            last_check: Utc::now(),
            response_time_ms: None,
            metadata: HashMap::new(),
        }
    }

    /// Add response time
    pub fn with_response_time(mut self, ms: u64) -> Self {
        self.response_time_ms = Some(ms);
        self
    }

    /// Add metadata entry
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Check if component is healthy
    pub fn is_healthy(&self) -> bool {
        self.status == ProbeStatus::Up
    }

    /// Check if component is operational (up or degraded)
    pub fn is_operational(&self) -> bool {
        matches!(self.status, ProbeStatus::Up | ProbeStatus::Degraded)
    }
}

/// A complete health signal containing system-wide status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthSignal {
    /// Overall system status
    pub system_status: SystemStatus,
    /// Individual component health
    pub components: Vec<ComponentHealth>,
    /// Signal timestamp
    pub timestamp: DateTime<Utc>,
    /// System uptime in seconds
    pub uptime_seconds: u64,
    /// Version information
    pub version: String,
}

impl HealthSignal {
    /// Create a new health signal
    pub fn new(components: Vec<ComponentHealth>, uptime_seconds: u64) -> Self {
        let system_status = SystemStatus::aggregate(&components);
        Self {
            system_status,
            components,
            timestamp: Utc::now(),
            uptime_seconds,
            version: crate::CNS_VERSION.to_string(),
        }
    }

    /// Get health for a specific component type
    pub fn get_component(&self, component_type: ComponentType) -> Option<&ComponentHealth> {
        self.components
            .iter()
            .find(|c| c.component_type == component_type)
    }

    /// Calculate overall health score (0.0 to 1.0)
    pub fn health_score(&self) -> f64 {
        if self.components.is_empty() {
            return 0.0;
        }

        let total_score: f64 = self.components.iter().map(|c| c.status.score()).sum();
        total_score / self.components.len() as f64
    }

    /// Get list of unhealthy components
    pub fn unhealthy_components(&self) -> Vec<&ComponentHealth> {
        self.components.iter().filter(|c| !c.is_healthy()).collect()
    }

    /// Get list of critical (down) components
    pub fn critical_components(&self) -> Vec<&ComponentHealth> {
        self.components
            .iter()
            .filter(|c| c.status == ProbeStatus::Down)
            .collect()
    }
}

/// Health check request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckRequest {
    /// Optional filter for specific components
    pub components: Option<Vec<ComponentType>>,
    /// Include detailed metadata
    pub include_metadata: bool,
}

impl Default for HealthCheckRequest {
    fn default() -> Self {
        Self {
            components: None,
            include_metadata: true,
        }
    }
}

/// Health check response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckResponse {
    /// The health signal
    pub signal: HealthSignal,
    /// Request processing time in milliseconds
    pub processing_time_ms: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_status_aggregate() {
        let healthy = vec![
            ComponentHealth::healthy(ComponentType::ForwardService),
            ComponentHealth::healthy(ComponentType::BackwardService),
        ];
        assert_eq!(SystemStatus::aggregate(&healthy), SystemStatus::Healthy);

        let degraded = vec![
            ComponentHealth::healthy(ComponentType::ForwardService),
            ComponentHealth::degraded(ComponentType::BackwardService, "slow"),
        ];
        assert_eq!(SystemStatus::aggregate(&degraded), SystemStatus::Degraded);

        let critical = vec![
            ComponentHealth::healthy(ComponentType::ForwardService),
            ComponentHealth::unhealthy(ComponentType::BackwardService, "down"),
        ];
        assert_eq!(SystemStatus::aggregate(&critical), SystemStatus::Critical);
    }

    #[test]
    fn test_probe_status_score() {
        assert_eq!(ProbeStatus::Up.score(), 1.0);
        assert_eq!(ProbeStatus::Degraded.score(), 0.5);
        assert_eq!(ProbeStatus::Down.score(), 0.0);
        assert_eq!(ProbeStatus::Unknown.score(), 0.25);
    }

    #[test]
    fn test_health_signal_score() {
        let components = vec![
            ComponentHealth::healthy(ComponentType::ForwardService),
            ComponentHealth::healthy(ComponentType::BackwardService),
            ComponentHealth::degraded(ComponentType::Redis, "high latency"),
            ComponentHealth::unhealthy(ComponentType::Qdrant, "unreachable"),
        ];

        let signal = HealthSignal::new(components, 3600);
        // (1.0 + 1.0 + 0.5 + 0.0) / 4 = 0.625
        assert_eq!(signal.health_score(), 0.625);
    }

    #[test]
    fn test_component_health_builder() {
        let health = ComponentHealth::healthy(ComponentType::ForwardService)
            .with_response_time(42)
            .with_metadata("region", "us-east-1")
            .with_metadata("version", "0.1.0");

        assert_eq!(health.response_time_ms, Some(42));
        assert_eq!(health.metadata.get("region").unwrap(), "us-east-1");
        assert_eq!(health.metadata.get("version").unwrap(), "0.1.0");
    }
}
