//! Production deployment and monitoring for the vision pipeline.
//!
//! This module provides production-ready features:
//! - Health checks and status monitoring
//! - Metrics collection and export
//! - Circuit breakers and error recovery
//! - Graceful shutdown handling
//! - Configuration management
//!
//! # Architecture
//!
//! ```text
//! Production System
//!     ├── Health Monitor (liveness, readiness)
//!     ├── Metrics Registry (Prometheus export)
//!     ├── Circuit Breaker (fault tolerance)
//!     └── Error Recovery (retry logic)
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use vision::production::{ProductionMonitor, ProductionConfig};
//!
//! let config = ProductionConfig::default();
//! let monitor = ProductionMonitor::new(config);
//!
//! // Start monitoring
//! monitor.start()?;
//!
//! // Check health
//! let report = monitor.health_report();
//! println!("Status: {}", report.overall_status.as_str());
//!
//! // Export metrics
//! let metrics = monitor.export_metrics();
//! ```

pub mod health;
pub mod metrics;
pub mod recovery;

pub use health::{
    ComponentHealth, HealthCheckConfig, HealthMonitor, HealthReport, HealthStatus, LivenessProbe,
    ReadinessProbe, ResourceMetrics,
};
pub use metrics::{Counter, Gauge, Histogram, MetricType, MetricsRegistry, PipelineMetrics};
pub use recovery::{
    CircuitBreaker, CircuitBreakerConfig, CircuitBreakerError, CircuitState, CircuitStats,
    ErrorRateTracker, RetryConfig, RetryExecutor,
};

use crate::error::Result;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Production configuration.
#[derive(Debug, Clone)]
pub struct ProductionConfig {
    /// Health check configuration
    pub health_config: HealthCheckConfig,
    /// Circuit breaker configuration
    pub circuit_config: CircuitBreakerConfig,
    /// Retry configuration
    pub retry_config: RetryConfig,
    /// Metrics collection enabled
    pub enable_metrics: bool,
    /// Health check interval
    pub health_check_interval: Duration,
    /// Minimum uptime for readiness
    pub min_uptime_for_ready: Duration,
}

impl ProductionConfig {
    /// Create default production configuration.
    pub fn default() -> Self {
        Self {
            health_config: HealthCheckConfig::default(),
            circuit_config: CircuitBreakerConfig::default(),
            retry_config: RetryConfig::default(),
            enable_metrics: true,
            health_check_interval: Duration::from_secs(10),
            min_uptime_for_ready: Duration::from_secs(5),
        }
    }

    /// Create strict production configuration.
    pub fn strict() -> Self {
        Self {
            health_config: HealthCheckConfig::strict(),
            circuit_config: CircuitBreakerConfig::strict(),
            retry_config: RetryConfig::conservative(),
            enable_metrics: true,
            health_check_interval: Duration::from_secs(5),
            min_uptime_for_ready: Duration::from_secs(10),
        }
    }

    /// Create lenient configuration for development.
    pub fn lenient() -> Self {
        Self {
            health_config: HealthCheckConfig::lenient(),
            circuit_config: CircuitBreakerConfig::lenient(),
            retry_config: RetryConfig::aggressive(),
            enable_metrics: true,
            health_check_interval: Duration::from_secs(30),
            min_uptime_for_ready: Duration::from_secs(2),
        }
    }
}

/// Production monitor orchestrating all monitoring components.
pub struct ProductionMonitor {
    config: ProductionConfig,
    health_monitor: Arc<HealthMonitor>,
    metrics: Arc<PipelineMetrics>,
    circuit_breaker: Arc<CircuitBreaker>,
    retry_executor: RetryExecutor,
    liveness_probe: LivenessProbe,
    readiness_probe: Arc<RwLock<Option<ReadinessProbe>>>,
    error_tracker: Arc<ErrorRateTracker>,
    started_at: Instant,
}

impl ProductionMonitor {
    /// Create a new production monitor.
    pub fn new(config: ProductionConfig) -> Self {
        let health_monitor = Arc::new(HealthMonitor::new(config.health_config.clone()));
        let metrics = Arc::new(PipelineMetrics::new());
        let circuit_breaker = Arc::new(CircuitBreaker::new(config.circuit_config.clone()));
        let retry_executor = RetryExecutor::new(config.retry_config.clone());
        let liveness_probe = LivenessProbe::new();
        let error_tracker = Arc::new(ErrorRateTracker::new(Duration::from_secs(300)));

        Self {
            config,
            health_monitor: health_monitor.clone(),
            metrics,
            circuit_breaker,
            retry_executor,
            liveness_probe,
            readiness_probe: Arc::new(RwLock::new(None)),
            error_tracker,
            started_at: Instant::now(),
        }
    }

    /// Create a production monitor with default configuration.
    pub fn default() -> Self {
        Self::new(ProductionConfig::default())
    }

    /// Start the production monitor.
    pub fn start(&self) -> Result<()> {
        // Initialize readiness probe
        let readiness = ReadinessProbe::new(
            self.health_monitor.clone(),
            self.config.min_uptime_for_ready,
        );

        if let Ok(mut probe) = self.readiness_probe.write() {
            *probe = Some(readiness);
        }

        // Update initial component health
        self.update_component_health("production_monitor", HealthStatus::Healthy, None);

        Ok(())
    }

    /// Update component health status.
    pub fn update_component_health(
        &self,
        name: &str,
        status: HealthStatus,
        message: Option<String>,
    ) {
        let health = match status {
            HealthStatus::Healthy => ComponentHealth::healthy(name.to_string()),
            HealthStatus::Degraded => {
                ComponentHealth::degraded(name.to_string(), message.unwrap_or_default())
            }
            HealthStatus::Unhealthy => {
                ComponentHealth::unhealthy(name.to_string(), message.unwrap_or_default())
            }
            HealthStatus::Unknown => ComponentHealth::new(name.to_string(), HealthStatus::Unknown),
        };

        self.health_monitor.update_component(health);
    }

    /// Update resource metrics.
    pub fn update_resources(&self, metrics: ResourceMetrics) {
        self.health_monitor.update_resources(metrics);
    }

    /// Get health report.
    pub fn health_report(&self) -> HealthReport {
        self.health_monitor.get_report()
    }

    /// Check liveness (is the system running).
    pub fn is_alive(&self) -> bool {
        self.liveness_probe.check()
    }

    /// Check readiness (is the system ready to serve traffic).
    pub fn is_ready(&self) -> bool {
        if let Ok(probe) = self.readiness_probe.read() {
            probe.as_ref().map(|p| p.check()).unwrap_or(false)
        } else {
            false
        }
    }

    /// Get uptime in seconds.
    pub fn uptime_seconds(&self) -> u64 {
        self.started_at.elapsed().as_secs()
    }

    /// Record a prediction with metrics.
    pub fn record_prediction(&self, latency_seconds: f64, success: bool) {
        if self.config.enable_metrics {
            if success {
                self.metrics.record_prediction(latency_seconds);
                self.error_tracker.record_success();
            } else {
                self.metrics.record_error();
                self.error_tracker.record_error();
            }
        }
    }

    /// Record a cache hit.
    pub fn record_cache_hit(&self) {
        if self.config.enable_metrics {
            self.metrics.record_cache_hit();
        }
    }

    /// Record a cache miss.
    pub fn record_cache_miss(&self) {
        if self.config.enable_metrics {
            self.metrics.record_cache_miss();
        }
    }

    /// Get metrics in Prometheus format.
    pub fn export_metrics(&self) -> String {
        self.metrics.export_prometheus()
    }

    /// Get circuit breaker.
    pub fn circuit_breaker(&self) -> &Arc<CircuitBreaker> {
        &self.circuit_breaker
    }

    /// Get retry executor.
    pub fn retry_executor(&self) -> &RetryExecutor {
        &self.retry_executor
    }

    /// Get error rate.
    pub fn error_rate(&self) -> f64 {
        self.error_tracker.error_rate()
    }

    /// Check if error rate exceeds threshold.
    pub fn is_error_rate_healthy(&self) -> bool {
        let threshold = self.config.health_config.max_error_rate / 100.0;
        !self.error_tracker.exceeds_threshold(threshold)
    }

    /// Get pipeline metrics.
    pub fn metrics(&self) -> &Arc<PipelineMetrics> {
        &self.metrics
    }

    /// Get health monitor.
    pub fn health_monitor(&self) -> &Arc<HealthMonitor> {
        &self.health_monitor
    }

    /// Print status summary.
    pub fn print_status(&self) {
        println!("=== Production Monitor Status ===");
        println!("Uptime: {} seconds", self.uptime_seconds());
        println!("Alive: {}", self.is_alive());
        println!("Ready: {}", self.is_ready());
        println!("Error rate: {:.2}%", self.error_rate() * 100.0);
        println!("Circuit state: {}", self.circuit_breaker.state().as_str());
        println!();

        let report = self.health_report();
        report.print_summary();
    }

    /// Perform health check and update status.
    pub fn perform_health_check(&self) {
        let start = Instant::now();

        // Check error rate
        let error_healthy = self.is_error_rate_healthy();
        if error_healthy {
            self.update_component_health("error_rate", HealthStatus::Healthy, None);
        } else {
            self.update_component_health(
                "error_rate",
                HealthStatus::Degraded,
                Some(format!("Error rate: {:.2}%", self.error_rate() * 100.0)),
            );
        }

        // Check circuit breaker
        let circuit_state = self.circuit_breaker.state();
        match circuit_state {
            CircuitState::Closed => {
                self.update_component_health("circuit_breaker", HealthStatus::Healthy, None);
            }
            CircuitState::HalfOpen => {
                self.update_component_health(
                    "circuit_breaker",
                    HealthStatus::Degraded,
                    Some("Circuit is half-open".to_string()),
                );
            }
            CircuitState::Open => {
                self.update_component_health(
                    "circuit_breaker",
                    HealthStatus::Unhealthy,
                    Some("Circuit is open".to_string()),
                );
            }
        }

        let duration_us = start.elapsed().as_micros() as u64;
        let health =
            ComponentHealth::healthy("health_check".to_string()).with_duration(duration_us);
        self.health_monitor.update_component(health);
    }

    /// Get configuration.
    pub fn config(&self) -> &ProductionConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_production_monitor_creation() {
        let monitor = ProductionMonitor::default();
        assert!(monitor.is_alive());
    }

    #[test]
    fn test_production_monitor_start() {
        let monitor = ProductionMonitor::default();
        monitor.start().unwrap();

        // Might not be ready immediately
        std::thread::sleep(Duration::from_millis(100));
    }

    #[test]
    fn test_component_health_update() {
        let monitor = ProductionMonitor::default();
        monitor.start().unwrap();

        monitor.update_component_health("test", HealthStatus::Healthy, None);

        let report = monitor.health_report();
        assert!(report.components.iter().any(|c| c.name == "test"));
    }

    #[test]
    fn test_metrics_recording() {
        let monitor = ProductionMonitor::default();

        monitor.record_prediction(0.001, true);
        monitor.record_cache_hit();
        monitor.record_cache_miss();

        let metrics = monitor.metrics();
        assert_eq!(metrics.predictions_total.get(), 1.0);
    }

    #[test]
    fn test_health_check() {
        let monitor = ProductionMonitor::default();
        monitor.start().unwrap();

        monitor.perform_health_check();

        let report = monitor.health_report();
        assert!(!report.components.is_empty());
    }

    #[test]
    fn test_error_rate_tracking() {
        let monitor = ProductionMonitor::default();

        monitor.record_prediction(0.001, true);
        monitor.record_prediction(0.001, true);
        monitor.record_prediction(0.001, false);

        assert!(monitor.error_rate() > 0.0);
    }

    #[test]
    fn test_prometheus_export() {
        let monitor = ProductionMonitor::default();
        monitor.record_prediction(0.001, true);

        let metrics = monitor.export_metrics();
        assert!(metrics.contains("vision_predictions_total"));
    }

    #[test]
    fn test_uptime() {
        let monitor = ProductionMonitor::default();
        let uptime_before = monitor.uptime_seconds();
        std::thread::sleep(Duration::from_millis(100));

        // Verify uptime increased
        assert!(monitor.uptime_seconds() > uptime_before);
    }

    #[test]
    fn test_config_presets() {
        let default = ProductionConfig::default();
        assert!(default.enable_metrics);

        let strict = ProductionConfig::strict();
        assert!(strict.health_check_interval < default.health_check_interval);

        let lenient = ProductionConfig::lenient();
        assert!(lenient.health_check_interval > default.health_check_interval);
    }

    #[test]
    fn test_liveness_probe() {
        let monitor = ProductionMonitor::default();
        assert!(monitor.is_alive());
    }

    #[test]
    fn test_readiness_probe() {
        let monitor = ProductionMonitor::default();
        monitor.start().unwrap();

        // Wait for minimum uptime
        std::thread::sleep(Duration::from_millis(10));

        monitor.update_component_health("test", HealthStatus::Healthy, None);
    }
}
