//! Health check and status monitoring for production deployment.
//!
//! This module provides comprehensive health monitoring:
//! - Component health checks
//! - System resource monitoring
//! - Dependency health tracking
//! - Aggregated health status
//! - Readiness and liveness probes

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Health status of a component.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    /// Component is healthy and operational
    Healthy,
    /// Component is degraded but still functional
    Degraded,
    /// Component is unhealthy and not operational
    Unhealthy,
    /// Component status is unknown
    Unknown,
}

impl HealthStatus {
    /// Check if status is healthy.
    pub fn is_healthy(&self) -> bool {
        matches!(self, HealthStatus::Healthy)
    }

    /// Check if status is operational (healthy or degraded).
    pub fn is_operational(&self) -> bool {
        matches!(self, HealthStatus::Healthy | HealthStatus::Degraded)
    }

    /// Convert to HTTP status code.
    pub fn to_http_code(&self) -> u16 {
        match self {
            HealthStatus::Healthy => 200,
            HealthStatus::Degraded => 200,
            HealthStatus::Unhealthy => 503,
            HealthStatus::Unknown => 503,
        }
    }

    /// Convert to string.
    pub fn as_str(&self) -> &str {
        match self {
            HealthStatus::Healthy => "healthy",
            HealthStatus::Degraded => "degraded",
            HealthStatus::Unhealthy => "unhealthy",
            HealthStatus::Unknown => "unknown",
        }
    }
}

/// Component health check result.
#[derive(Debug, Clone)]
pub struct ComponentHealth {
    pub name: String,
    pub status: HealthStatus,
    pub message: Option<String>,
    pub last_check: Instant,
    pub check_duration_us: u64,
}

impl ComponentHealth {
    /// Create a new component health result.
    pub fn new(name: String, status: HealthStatus) -> Self {
        Self {
            name,
            status,
            message: None,
            last_check: Instant::now(),
            check_duration_us: 0,
        }
    }

    /// Create a healthy status.
    pub fn healthy(name: String) -> Self {
        Self::new(name, HealthStatus::Healthy)
    }

    /// Create a degraded status with message.
    pub fn degraded(name: String, message: String) -> Self {
        Self {
            name,
            status: HealthStatus::Degraded,
            message: Some(message),
            last_check: Instant::now(),
            check_duration_us: 0,
        }
    }

    /// Create an unhealthy status with message.
    pub fn unhealthy(name: String, message: String) -> Self {
        Self {
            name,
            status: HealthStatus::Unhealthy,
            message: Some(message),
            last_check: Instant::now(),
            check_duration_us: 0,
        }
    }

    /// Set the check duration.
    pub fn with_duration(mut self, duration_us: u64) -> Self {
        self.check_duration_us = duration_us;
        self
    }

    /// Check if this component is healthy.
    pub fn is_healthy(&self) -> bool {
        self.status.is_healthy()
    }
}

/// System resource metrics.
#[derive(Debug, Clone)]
pub struct ResourceMetrics {
    pub timestamp: Instant,
    pub memory_used_mb: f64,
    pub memory_available_mb: f64,
    pub cpu_usage_percent: f64,
    pub thread_count: usize,
}

impl ResourceMetrics {
    /// Create new resource metrics.
    pub fn new() -> Self {
        Self {
            timestamp: Instant::now(),
            memory_used_mb: 0.0,
            memory_available_mb: 0.0,
            cpu_usage_percent: 0.0,
            thread_count: 0,
        }
    }

    /// Get memory usage percentage.
    pub fn memory_usage_percent(&self) -> f64 {
        if self.memory_available_mb > 0.0 {
            (self.memory_used_mb / self.memory_available_mb) * 100.0
        } else {
            0.0
        }
    }

    /// Check if memory usage is within threshold.
    pub fn is_memory_healthy(&self, threshold_percent: f64) -> bool {
        self.memory_usage_percent() < threshold_percent
    }

    /// Check if CPU usage is within threshold.
    pub fn is_cpu_healthy(&self, threshold_percent: f64) -> bool {
        self.cpu_usage_percent < threshold_percent
    }
}

impl Default for ResourceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Health check configuration.
#[derive(Debug, Clone)]
pub struct HealthCheckConfig {
    /// Memory usage threshold (percent)
    pub memory_threshold_percent: f64,
    /// CPU usage threshold (percent)
    pub cpu_threshold_percent: f64,
    /// Maximum allowed latency (ms)
    pub max_latency_ms: f64,
    /// Minimum cache hit rate (percent)
    pub min_cache_hit_rate: f64,
    /// Maximum error rate (percent)
    pub max_error_rate: f64,
}

impl HealthCheckConfig {
    /// Create default configuration.
    pub fn default() -> Self {
        Self {
            memory_threshold_percent: 85.0,
            cpu_threshold_percent: 80.0,
            max_latency_ms: 100.0,
            min_cache_hit_rate: 70.0,
            max_error_rate: 5.0,
        }
    }

    /// Create strict configuration for production.
    pub fn strict() -> Self {
        Self {
            memory_threshold_percent: 75.0,
            cpu_threshold_percent: 70.0,
            max_latency_ms: 50.0,
            min_cache_hit_rate: 80.0,
            max_error_rate: 1.0,
        }
    }

    /// Create lenient configuration for development.
    pub fn lenient() -> Self {
        Self {
            memory_threshold_percent: 95.0,
            cpu_threshold_percent: 90.0,
            max_latency_ms: 500.0,
            min_cache_hit_rate: 50.0,
            max_error_rate: 10.0,
        }
    }
}

/// Aggregated health report.
#[derive(Debug, Clone)]
pub struct HealthReport {
    pub overall_status: HealthStatus,
    pub components: Vec<ComponentHealth>,
    pub resources: ResourceMetrics,
    pub timestamp: Instant,
}

impl HealthReport {
    /// Create a new health report.
    pub fn new(components: Vec<ComponentHealth>, resources: ResourceMetrics) -> Self {
        let overall_status = Self::calculate_overall_status(&components);
        Self {
            overall_status,
            components,
            resources,
            timestamp: Instant::now(),
        }
    }

    /// Calculate overall status from component statuses.
    fn calculate_overall_status(components: &[ComponentHealth]) -> HealthStatus {
        if components.is_empty() {
            return HealthStatus::Unknown;
        }

        let has_unhealthy = components
            .iter()
            .any(|c| c.status == HealthStatus::Unhealthy);
        let has_degraded = components
            .iter()
            .any(|c| c.status == HealthStatus::Degraded);

        if has_unhealthy {
            HealthStatus::Unhealthy
        } else if has_degraded {
            HealthStatus::Degraded
        } else {
            HealthStatus::Healthy
        }
    }

    /// Check if overall status is healthy.
    pub fn is_healthy(&self) -> bool {
        self.overall_status.is_healthy()
    }

    /// Get unhealthy components.
    pub fn unhealthy_components(&self) -> Vec<&ComponentHealth> {
        self.components
            .iter()
            .filter(|c| c.status == HealthStatus::Unhealthy)
            .collect()
    }

    /// Print summary.
    pub fn print_summary(&self) {
        println!("=== Health Report ===");
        println!("Overall Status: {}", self.overall_status.as_str());
        println!("Components:");
        for component in &self.components {
            let status_icon = match component.status {
                HealthStatus::Healthy => "✓",
                HealthStatus::Degraded => "⚠",
                HealthStatus::Unhealthy => "✗",
                HealthStatus::Unknown => "?",
            };
            print!(
                "  {} {} - {}",
                status_icon,
                component.name,
                component.status.as_str()
            );
            if let Some(msg) = &component.message {
                print!(" ({})", msg);
            }
            println!();
        }
        println!("Resources:");
        println!(
            "  Memory: {:.1}% ({:.1} MB / {:.1} MB)",
            self.resources.memory_usage_percent(),
            self.resources.memory_used_mb,
            self.resources.memory_available_mb
        );
        println!("  CPU: {:.1}%", self.resources.cpu_usage_percent);
        println!("  Threads: {}", self.resources.thread_count);
    }
}

/// Health monitor for tracking component and system health.
pub struct HealthMonitor {
    config: HealthCheckConfig,
    components: Arc<RwLock<HashMap<String, ComponentHealth>>>,
    resources: Arc<RwLock<ResourceMetrics>>,
}

impl HealthMonitor {
    /// Create a new health monitor.
    pub fn new(config: HealthCheckConfig) -> Self {
        Self {
            config,
            components: Arc::new(RwLock::new(HashMap::new())),
            resources: Arc::new(RwLock::new(ResourceMetrics::new())),
        }
    }

    /// Create a health monitor with default configuration.
    pub fn default() -> Self {
        Self::new(HealthCheckConfig::default())
    }

    /// Update component health.
    pub fn update_component(&self, health: ComponentHealth) {
        if let Ok(mut components) = self.components.write() {
            components.insert(health.name.clone(), health);
        }
    }

    /// Update resource metrics.
    pub fn update_resources(&self, metrics: ResourceMetrics) {
        if let Ok(mut resources) = self.resources.write() {
            *resources = metrics;
        }
    }

    /// Get current health report.
    pub fn get_report(&self) -> HealthReport {
        let components = self
            .components
            .read()
            .ok()
            .map(|c| c.values().cloned().collect())
            .unwrap_or_default();

        let resources = self
            .resources
            .read()
            .ok()
            .map(|r| r.clone())
            .unwrap_or_default();

        HealthReport::new(components, resources)
    }

    /// Check overall health.
    pub fn is_healthy(&self) -> bool {
        self.get_report().is_healthy()
    }

    /// Get configuration.
    pub fn config(&self) -> &HealthCheckConfig {
        &self.config
    }

    /// Remove a component from monitoring.
    pub fn remove_component(&self, name: &str) {
        if let Ok(mut components) = self.components.write() {
            components.remove(name);
        }
    }

    /// Clear all component health data.
    pub fn clear_components(&self) {
        if let Ok(mut components) = self.components.write() {
            components.clear();
        }
    }
}

/// Liveness probe for Kubernetes-style health checks.
///
/// Indicates whether the application is running.
pub struct LivenessProbe {
    started: Instant,
}

impl LivenessProbe {
    /// Create a new liveness probe.
    pub fn new() -> Self {
        Self {
            started: Instant::now(),
        }
    }

    /// Check liveness (always returns true if application is running).
    pub fn check(&self) -> bool {
        true
    }

    /// Get uptime in seconds.
    pub fn uptime_seconds(&self) -> u64 {
        self.started.elapsed().as_secs()
    }
}

impl Default for LivenessProbe {
    fn default() -> Self {
        Self::new()
    }
}

/// Readiness probe for Kubernetes-style health checks.
///
/// Indicates whether the application is ready to serve traffic.
pub struct ReadinessProbe {
    health_monitor: Arc<HealthMonitor>,
    min_uptime: Duration,
    started: Instant,
}

impl ReadinessProbe {
    /// Create a new readiness probe.
    pub fn new(health_monitor: Arc<HealthMonitor>, min_uptime: Duration) -> Self {
        Self {
            health_monitor,
            min_uptime,
            started: Instant::now(),
        }
    }

    /// Check readiness.
    pub fn check(&self) -> bool {
        // Check minimum uptime
        if self.started.elapsed() < self.min_uptime {
            return false;
        }

        // Check overall health
        self.health_monitor.is_healthy()
    }

    /// Get time until ready (if not ready yet).
    pub fn time_until_ready(&self) -> Option<Duration> {
        let elapsed = self.started.elapsed();
        if elapsed < self.min_uptime {
            Some(self.min_uptime - elapsed)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_status() {
        assert!(HealthStatus::Healthy.is_healthy());
        assert!(!HealthStatus::Degraded.is_healthy());
        assert!(HealthStatus::Degraded.is_operational());
        assert!(!HealthStatus::Unhealthy.is_operational());
    }

    #[test]
    fn test_component_health() {
        let health = ComponentHealth::healthy("test".to_string());
        assert!(health.is_healthy());
        assert_eq!(health.status, HealthStatus::Healthy);

        let degraded = ComponentHealth::degraded("test".to_string(), "slow".to_string());
        assert!(!degraded.is_healthy());
        assert_eq!(degraded.status, HealthStatus::Degraded);
    }

    #[test]
    fn test_resource_metrics() {
        let metrics = ResourceMetrics {
            timestamp: Instant::now(),
            memory_used_mb: 500.0,
            memory_available_mb: 1000.0,
            cpu_usage_percent: 60.0,
            thread_count: 8,
        };

        assert_eq!(metrics.memory_usage_percent(), 50.0);
        assert!(metrics.is_memory_healthy(80.0));
        assert!(!metrics.is_memory_healthy(40.0));
        assert!(metrics.is_cpu_healthy(70.0));
    }

    #[test]
    fn test_health_report() {
        let components = vec![
            ComponentHealth::healthy("component1".to_string()),
            ComponentHealth::healthy("component2".to_string()),
        ];
        let resources = ResourceMetrics::new();
        let report = HealthReport::new(components, resources);

        assert!(report.is_healthy());
        assert_eq!(report.overall_status, HealthStatus::Healthy);
    }

    #[test]
    fn test_health_report_degraded() {
        let components = vec![
            ComponentHealth::healthy("component1".to_string()),
            ComponentHealth::degraded("component2".to_string(), "slow".to_string()),
        ];
        let resources = ResourceMetrics::new();
        let report = HealthReport::new(components, resources);

        assert!(!report.is_healthy());
        assert_eq!(report.overall_status, HealthStatus::Degraded);
    }

    #[test]
    fn test_health_report_unhealthy() {
        let components = vec![
            ComponentHealth::healthy("component1".to_string()),
            ComponentHealth::unhealthy("component2".to_string(), "failed".to_string()),
        ];
        let resources = ResourceMetrics::new();
        let report = HealthReport::new(components, resources);

        assert!(!report.is_healthy());
        assert_eq!(report.overall_status, HealthStatus::Unhealthy);
        assert_eq!(report.unhealthy_components().len(), 1);
    }

    #[test]
    fn test_health_monitor() {
        let monitor = HealthMonitor::default();

        monitor.update_component(ComponentHealth::healthy("test".to_string()));

        let report = monitor.get_report();
        assert_eq!(report.components.len(), 1);
        assert!(report.is_healthy());
    }

    #[test]
    fn test_liveness_probe() {
        let probe = LivenessProbe::new();
        assert!(probe.check());
        // Just verify the call succeeds (uptime is always >= 0 by type)
        let _uptime = probe.uptime_seconds();
    }

    #[test]
    fn test_readiness_probe() {
        let monitor = Arc::new(HealthMonitor::default());
        monitor.update_component(ComponentHealth::healthy("test".to_string()));

        let probe = ReadinessProbe::new(monitor, Duration::from_millis(10));

        // Might not be ready yet due to min uptime
        std::thread::sleep(Duration::from_millis(20));
        assert!(probe.check());
    }

    #[test]
    fn test_health_config_presets() {
        let default = HealthCheckConfig::default();
        assert_eq!(default.memory_threshold_percent, 85.0);

        let strict = HealthCheckConfig::strict();
        assert!(strict.memory_threshold_percent < default.memory_threshold_percent);

        let lenient = HealthCheckConfig::lenient();
        assert!(lenient.memory_threshold_percent > default.memory_threshold_percent);
    }

    #[test]
    fn test_component_removal() {
        let monitor = HealthMonitor::default();
        monitor.update_component(ComponentHealth::healthy("test".to_string()));

        assert_eq!(monitor.get_report().components.len(), 1);

        monitor.remove_component("test");
        assert_eq!(monitor.get_report().components.len(), 0);
    }
}
