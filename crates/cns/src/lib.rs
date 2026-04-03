//! # JANUS Central Nervous System (CNS)
//!
//! The CNS module provides comprehensive health monitoring and metrics collection
//! for the JANUS trading system, inspired by the biological Central Nervous System.
//!
//! ## Architecture
//!
//! - **Brain (Coordinator)**: Aggregates health from all subsystems
//! - **Spinal Cord (Metrics Pipeline)**: Prometheus metrics collection and export
//! - **Sensory Input (Probes)**: Individual service health checks
//! - **Motor Output (Actions)**: Alerting and auto-recovery
//!
//! ## Components
//!
//! - `brain`: Main health coordinator and aggregator
//! - `metrics`: Prometheus metrics definitions and collectors
//! - `probes`: Health check probes for services and dependencies
//! - `reflexes`: Auto-recovery actions and circuit breakers
//! - `signals`: Health signal types and status definitions

pub mod alerts;
pub mod brain;
pub mod metrics;
pub mod neuromorphic;
pub mod preflight;
pub mod probes;
pub mod reflexes;
pub mod restart;
pub mod shutdown;
pub mod signals;
pub mod watchdog;

// Re-exports for convenience
pub use alerts::{AlertConfig, AlertManager, AlertResult};
pub use brain::{Brain, BrainConfig};
pub use metrics::{CNSMetrics, MetricsRegistry};
pub use neuromorphic::{BrainHealthSummary, BrainRegionStatus, NeuromorphicBrain};
pub use preflight::{
    BootPhase, BootReport, CheckOutcome, CheckResult, Criticality, PreFlightCheck, PreFlightConfig,
    PreFlightRunner,
};
pub use probes::{HealthProbe, ProbeResult};
pub use reflexes::{CircuitBreaker, RefexAction, Reflex};
pub use restart::{RestartConfig, RestartManager, RestartResult, RestartStrategy};
pub use shutdown::{ShutdownConfig, ShutdownCoordinator, ShutdownReport, ShutdownSignal};
pub use signals::ProbeStatus;
pub use signals::{
    ComponentHealth, HealthCheckRequest, HealthCheckResponse, HealthSignal, SystemStatus,
};
pub use watchdog::{
    CnsWatchdog, ComponentCriticality, ComponentRegistration, ComponentState, KillSwitch,
    NoOpKillSwitch, WatchdogConfig, WatchdogEvent, WatchdogHandle, WatchdogMetrics,
    WatchdogSnapshot,
};

use thiserror::Error;

#[derive(Error, Debug)]
pub enum CNSError {
    #[error("Health probe failed: {0}")]
    ProbeFailure(String),

    #[error("Metrics collection error: {0}")]
    MetricsError(String),

    #[error("Component unhealthy: {component} - {reason}")]
    ComponentUnhealthy { component: String, reason: String },

    #[error("Circuit breaker open for: {0}")]
    CircuitBreakerOpen(String),

    #[error("Communication error: {0}")]
    CommunicationError(#[from] std::io::Error),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Internal error: {0}")]
    InternalError(#[from] anyhow::Error),
}

pub type Result<T> = std::result::Result<T, CNSError>;

/// CNS System version
pub const CNS_VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!CNS_VERSION.is_empty());
    }
}
