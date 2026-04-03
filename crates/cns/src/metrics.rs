//! # Prometheus Metrics Module
//!
//! Defines Prometheus metrics for monitoring the JANUS CNS health system.
//! These metrics expose the "vital signs" of the trading system.

use prometheus::{
    Gauge, GaugeVec, Histogram, HistogramOpts, HistogramVec, IntCounter, IntCounterVec, IntGauge,
    IntGaugeVec, Opts, Registry,
};
use std::sync::{Arc, LazyLock};

use crate::signals::{ComponentType, HealthSignal, SystemStatus};

/// Global metrics registry
pub static METRICS_REGISTRY: LazyLock<Arc<MetricsRegistry>> =
    LazyLock::new(|| Arc::new(MetricsRegistry::new()));

/// Centralized metrics registry for CNS
pub struct MetricsRegistry {
    /// Prometheus registry
    pub registry: Registry,

    // === System-Level Metrics ===
    /// Overall system health score (0.0 to 1.0)
    pub system_health_score: Gauge,

    /// System status (0=Starting, 1=Healthy, 2=Degraded, 3=Critical, 4=Shutdown)
    pub system_status: IntGauge,

    /// System uptime in seconds
    pub system_uptime_seconds: IntGauge,

    // === Component Health Metrics ===
    /// Component health status by type (0=Down, 0.25=Unknown, 0.5=Degraded, 1.0=Up)
    pub component_health: GaugeVec,

    /// Component response time in milliseconds
    pub component_response_time_ms: GaugeVec,

    /// Component probe execution count
    pub component_probe_total: IntCounterVec,

    /// Component probe failures
    pub component_probe_failures: IntCounterVec,

    // === Service-Specific Metrics ===
    /// Forward service (Wake State) metrics
    pub forward_active_engines: IntGauge,
    pub forward_orders_submitted: IntCounter,
    pub forward_orders_filled: IntCounter,
    pub forward_orders_rejected: IntCounter,

    /// Backward service (Sleep State) metrics
    pub backward_training_iterations: IntCounter,
    pub backward_memory_consolidations: IntCounter,
    pub backward_regime_updates: IntCounter,

    /// Gateway service metrics
    pub gateway_http_requests: IntCounterVec,
    pub gateway_http_request_duration: HistogramVec,

    // === Dependency Metrics ===
    /// Redis connection pool metrics
    pub redis_connections_active: IntGauge,
    pub redis_commands_total: IntCounterVec,
    pub redis_command_duration: HistogramVec,

    /// Qdrant vector database metrics
    pub qdrant_vectors_stored: IntGauge,
    pub qdrant_search_requests: IntCounter,
    pub qdrant_search_duration: Histogram,

    // === Communication Channel Metrics ===
    /// Shared memory IPC metrics
    pub shm_messages_sent: IntCounter,
    pub shm_messages_received: IntCounter,
    pub shm_message_size_bytes: Histogram,

    /// gRPC channel metrics
    pub grpc_requests_total: IntCounterVec,
    pub grpc_request_duration: HistogramVec,
    pub grpc_active_connections: IntGauge,

    // === Exchange Data Metrics ===
    /// Total messages received from exchanges
    pub exchange_message_total: IntCounterVec,

    /// Exchange message parse errors
    pub exchange_message_parse_errors: IntCounterVec,

    /// Exchange connection health status (1.0=Healthy, 0.5=Degraded, 0.0=Down)
    pub exchange_health_status: GaugeVec,

    /// Exchange message latency
    pub exchange_latency_seconds: HistogramVec,

    // === Circuit Breaker Metrics ===
    /// Circuit breaker state (0=Closed, 1=Open, 2=HalfOpen)
    pub circuit_breaker_state: IntGaugeVec,

    /// Circuit breaker trip count
    pub circuit_breaker_trips: IntCounterVec,

    // === Resource Metrics ===
    /// Memory usage in bytes
    pub memory_usage_bytes: IntGauge,

    /// CPU usage percentage (0-100)
    pub cpu_usage_percent: Gauge,

    /// Active goroutines/tasks
    pub active_tasks: IntGauge,
}

impl MetricsRegistry {
    /// Create a new metrics registry with all CNS metrics
    pub fn new() -> Self {
        let registry = Registry::new();

        // System-Level Metrics
        let system_health_score = Gauge::with_opts(
            Opts::new(
                "janus_cns_system_health_score",
                "Overall system health score (0.0 to 1.0)",
            )
            .namespace("janus")
            .subsystem("cns"),
        )
        .expect("failed to create janus_cns_system_health_score gauge");

        let system_status = IntGauge::with_opts(
            Opts::new(
                "janus_cns_system_status",
                "System status (0=Starting, 1=Healthy, 2=Degraded, 3=Critical, 4=Shutdown)",
            )
            .namespace("janus")
            .subsystem("cns"),
        )
        .expect("failed to create janus_cns_system_status gauge");

        let system_uptime_seconds = IntGauge::with_opts(
            Opts::new(
                "janus_cns_system_uptime_seconds",
                "System uptime in seconds",
            )
            .namespace("janus")
            .subsystem("cns"),
        )
        .expect("failed to create janus_cns_system_uptime_seconds gauge");

        // Component Health Metrics
        let component_health = GaugeVec::new(
            Opts::new(
                "janus_cns_component_health",
                "Component health status score",
            )
            .namespace("janus")
            .subsystem("cns"),
            &["component"],
        )
        .expect("failed to create janus_cns_component_health gauge_vec");

        let component_response_time_ms = GaugeVec::new(
            Opts::new(
                "janus_cns_component_response_time_ms",
                "Component response time in milliseconds",
            )
            .namespace("janus")
            .subsystem("cns"),
            &["component"],
        )
        .expect("failed to create janus_cns_component_response_time_ms gauge_vec");

        let component_probe_total = IntCounterVec::new(
            Opts::new(
                "janus_cns_component_probe_total",
                "Total component probe executions",
            )
            .namespace("janus")
            .subsystem("cns"),
            &["component", "status"],
        )
        .expect("failed to create janus_cns_component_probe_total counter_vec");

        let component_probe_failures = IntCounterVec::new(
            Opts::new(
                "janus_cns_component_probe_failures",
                "Component probe failures",
            )
            .namespace("janus")
            .subsystem("cns"),
            &["component", "reason"],
        )
        .expect("failed to create janus_cns_component_probe_failures counter_vec");

        // Forward Service Metrics
        let forward_active_engines = IntGauge::with_opts(
            Opts::new(
                "janus_forward_active_engines",
                "Number of active trading engines",
            )
            .namespace("janus")
            .subsystem("forward"),
        )
        .expect("failed to create janus_forward_active_engines gauge");

        let forward_orders_submitted = IntCounter::with_opts(
            Opts::new(
                "janus_forward_orders_submitted_total",
                "Total orders submitted",
            )
            .namespace("janus")
            .subsystem("forward"),
        )
        .expect("failed to create janus_forward_orders_submitted_total counter");

        let forward_orders_filled = IntCounter::with_opts(
            Opts::new("janus_forward_orders_filled_total", "Total orders filled")
                .namespace("janus")
                .subsystem("forward"),
        )
        .expect("failed to create janus_forward_orders_filled_total counter");

        let forward_orders_rejected = IntCounter::with_opts(
            Opts::new(
                "janus_forward_orders_rejected_total",
                "Total orders rejected",
            )
            .namespace("janus")
            .subsystem("forward"),
        )
        .expect("failed to create janus_forward_orders_rejected_total counter");

        // Backward Service Metrics
        let backward_training_iterations = IntCounter::with_opts(
            Opts::new(
                "janus_backward_training_iterations_total",
                "Total training iterations",
            )
            .namespace("janus")
            .subsystem("backward"),
        )
        .expect("failed to create janus_backward_training_iterations_total counter");

        let backward_memory_consolidations = IntCounter::with_opts(
            Opts::new(
                "janus_backward_memory_consolidations_total",
                "Total memory consolidations",
            )
            .namespace("janus")
            .subsystem("backward"),
        )
        .expect("failed to create janus_backward_memory_consolidations_total counter");

        let backward_regime_updates = IntCounter::with_opts(
            Opts::new(
                "janus_backward_regime_updates_total",
                "Total regime updates",
            )
            .namespace("janus")
            .subsystem("backward"),
        )
        .expect("failed to create janus_backward_regime_updates_total counter");

        // Gateway Service Metrics
        let gateway_http_requests = IntCounterVec::new(
            Opts::new("janus_gateway_http_requests_total", "Total HTTP requests")
                .namespace("janus")
                .subsystem("gateway"),
            &["method", "endpoint", "status"],
        )
        .expect("failed to create janus_gateway_http_requests_total counter_vec");

        let gateway_http_request_duration = HistogramVec::new(
            HistogramOpts::new(
                "janus_gateway_http_request_duration_seconds",
                "HTTP request duration in seconds",
            )
            .namespace("janus")
            .subsystem("gateway")
            .buckets(vec![
                0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5,
            ]),
            &["method", "endpoint"],
        )
        .expect("failed to create janus_gateway_http_request_duration_seconds histogram_vec");

        // Redis Metrics
        let redis_connections_active = IntGauge::with_opts(
            Opts::new("janus_redis_connections_active", "Active Redis connections")
                .namespace("janus")
                .subsystem("redis"),
        )
        .expect("failed to create janus_redis_connections_active gauge");

        let redis_commands_total = IntCounterVec::new(
            Opts::new("janus_redis_commands_total", "Total Redis commands")
                .namespace("janus")
                .subsystem("redis"),
            &["command", "status"],
        )
        .expect("failed to create janus_redis_commands_total counter_vec");

        let redis_command_duration = HistogramVec::new(
            HistogramOpts::new(
                "janus_redis_command_duration_seconds",
                "Redis command duration in seconds",
            )
            .namespace("janus")
            .subsystem("redis")
            .buckets(vec![0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]),
            &["command"],
        )
        .expect("failed to create janus_redis_command_duration_seconds histogram_vec");

        // Qdrant Metrics
        let qdrant_vectors_stored = IntGauge::with_opts(
            Opts::new(
                "janus_qdrant_vectors_stored",
                "Number of vectors stored in Qdrant",
            )
            .namespace("janus")
            .subsystem("qdrant"),
        )
        .expect("failed to create janus_qdrant_vectors_stored gauge");

        let qdrant_search_requests = IntCounter::with_opts(
            Opts::new(
                "janus_qdrant_search_requests_total",
                "Total Qdrant search requests",
            )
            .namespace("janus")
            .subsystem("qdrant"),
        )
        .expect("failed to create janus_qdrant_search_requests_total counter");

        let qdrant_search_duration = Histogram::with_opts(
            HistogramOpts::new(
                "janus_qdrant_search_duration_seconds",
                "Qdrant search duration in seconds",
            )
            .namespace("janus")
            .subsystem("qdrant")
            .buckets(vec![0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]),
        )
        .expect("failed to create janus_qdrant_search_duration_seconds histogram");

        // Shared Memory Metrics
        let shm_messages_sent = IntCounter::with_opts(
            Opts::new(
                "janus_shm_messages_sent_total",
                "Total messages sent via shared memory",
            )
            .namespace("janus")
            .subsystem("shm"),
        )
        .expect("failed to create janus_shm_messages_sent_total counter");

        let shm_messages_received = IntCounter::with_opts(
            Opts::new(
                "janus_shm_messages_received_total",
                "Total messages received via shared memory",
            )
            .namespace("janus")
            .subsystem("shm"),
        )
        .expect("failed to create janus_shm_messages_received_total counter");

        let shm_message_size_bytes = Histogram::with_opts(
            HistogramOpts::new(
                "janus_shm_message_size_bytes",
                "Shared memory message size in bytes",
            )
            .namespace("janus")
            .subsystem("shm")
            .buckets(vec![1024.0, 4096.0, 16384.0, 65536.0, 262144.0, 1048576.0]),
        )
        .expect("failed to create janus_shm_message_size_bytes histogram");

        // gRPC Metrics
        let grpc_requests_total = IntCounterVec::new(
            Opts::new("janus_grpc_requests_total", "Total gRPC requests")
                .namespace("janus")
                .subsystem("grpc"),
            &["service", "method", "status"],
        )
        .expect("failed to create janus_grpc_requests_total counter_vec");

        let grpc_request_duration = HistogramVec::new(
            HistogramOpts::new(
                "janus_grpc_request_duration_seconds",
                "gRPC request duration in seconds",
            )
            .namespace("janus")
            .subsystem("grpc")
            .buckets(vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.5, 1.0]),
            &["service", "method"],
        )
        .expect("failed to create janus_grpc_request_duration_seconds histogram_vec");

        let grpc_active_connections = IntGauge::with_opts(
            Opts::new("janus_grpc_active_connections", "Active gRPC connections")
                .namespace("janus")
                .subsystem("grpc"),
        )
        .expect("failed to create janus_grpc_active_connections gauge");

        // Exchange Data Metrics
        let exchange_message_total = IntCounterVec::new(
            Opts::new(
                "janus_exchange_message_total",
                "Total messages received from exchanges",
            )
            .namespace("janus")
            .subsystem("exchange"),
            &["exchange", "channel", "symbol"],
        )
        .expect("failed to create janus_exchange_message_total counter_vec");

        let exchange_message_parse_errors = IntCounterVec::new(
            Opts::new(
                "janus_exchange_message_parse_errors_total",
                "Exchange message parse errors",
            )
            .namespace("janus")
            .subsystem("exchange"),
            &["exchange", "reason"],
        )
        .expect("failed to create janus_exchange_message_parse_errors_total counter_vec");

        let exchange_health_status = GaugeVec::new(
            Opts::new(
                "janus_exchange_health_status",
                "Exchange connection health status (1.0=Healthy, 0.5=Degraded, 0.0=Down)",
            )
            .namespace("janus")
            .subsystem("exchange"),
            &["exchange"],
        )
        .expect("failed to create janus_exchange_health_status gauge_vec");

        let exchange_latency_seconds = HistogramVec::new(
            HistogramOpts::new(
                "janus_exchange_latency_seconds",
                "Exchange message latency in seconds",
            )
            .namespace("janus")
            .subsystem("exchange")
            .buckets(vec![0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]),
            &["exchange", "channel"],
        )
        .expect("failed to create janus_exchange_latency_seconds histogram_vec");

        // Circuit Breaker Metrics
        let circuit_breaker_state = IntGaugeVec::new(
            Opts::new(
                "janus_circuit_breaker_state",
                "Circuit breaker state (0=Closed, 1=Open, 2=HalfOpen)",
            )
            .namespace("janus")
            .subsystem("circuit_breaker"),
            &["component"],
        )
        .expect("failed to create janus_circuit_breaker_state gauge_vec");

        let circuit_breaker_trips = IntCounterVec::new(
            Opts::new(
                "janus_circuit_breaker_trips_total",
                "Circuit breaker trip count",
            )
            .namespace("janus")
            .subsystem("circuit_breaker"),
            &["component", "reason"],
        )
        .expect("failed to create janus_circuit_breaker_trips_total counter_vec");

        // Resource Metrics
        let memory_usage_bytes = IntGauge::with_opts(
            Opts::new("janus_memory_usage_bytes", "Memory usage in bytes")
                .namespace("janus")
                .subsystem("resources"),
        )
        .expect("failed to create janus_memory_usage_bytes gauge");

        let cpu_usage_percent = Gauge::with_opts(
            Opts::new("janus_cpu_usage_percent", "CPU usage percentage")
                .namespace("janus")
                .subsystem("resources"),
        )
        .expect("failed to create janus_cpu_usage_percent gauge");

        let active_tasks = IntGauge::with_opts(
            Opts::new("janus_active_tasks", "Number of active tasks")
                .namespace("janus")
                .subsystem("resources"),
        )
        .expect("failed to create janus_active_tasks gauge");

        // Register all metrics
        registry
            .register(Box::new(system_health_score.clone()))
            .expect("failed to register system_health_score");
        registry
            .register(Box::new(system_status.clone()))
            .expect("failed to register system_status");
        registry
            .register(Box::new(system_uptime_seconds.clone()))
            .expect("failed to register system_uptime_seconds");
        registry
            .register(Box::new(component_health.clone()))
            .expect("failed to register component_health");
        registry
            .register(Box::new(component_response_time_ms.clone()))
            .expect("failed to register component_response_time_ms");
        registry
            .register(Box::new(component_probe_total.clone()))
            .expect("failed to register component_probe_total");
        registry
            .register(Box::new(component_probe_failures.clone()))
            .expect("failed to register component_probe_failures");
        registry
            .register(Box::new(forward_active_engines.clone()))
            .expect("failed to register forward_active_engines");
        registry
            .register(Box::new(forward_orders_submitted.clone()))
            .expect("failed to register forward_orders_submitted");
        registry
            .register(Box::new(forward_orders_filled.clone()))
            .expect("failed to register forward_orders_filled");
        registry
            .register(Box::new(forward_orders_rejected.clone()))
            .expect("failed to register forward_orders_rejected");
        registry
            .register(Box::new(backward_training_iterations.clone()))
            .expect("failed to register backward_training_iterations");
        registry
            .register(Box::new(backward_memory_consolidations.clone()))
            .expect("failed to register backward_memory_consolidations");
        registry
            .register(Box::new(backward_regime_updates.clone()))
            .expect("failed to register backward_regime_updates");
        registry
            .register(Box::new(gateway_http_requests.clone()))
            .expect("failed to register gateway_http_requests");
        registry
            .register(Box::new(gateway_http_request_duration.clone()))
            .expect("failed to register gateway_http_request_duration");
        registry
            .register(Box::new(redis_connections_active.clone()))
            .expect("failed to register redis_connections_active");
        registry
            .register(Box::new(redis_commands_total.clone()))
            .expect("failed to register redis_commands_total");
        registry
            .register(Box::new(redis_command_duration.clone()))
            .expect("failed to register redis_command_duration");
        registry
            .register(Box::new(qdrant_vectors_stored.clone()))
            .expect("failed to register qdrant_vectors_stored");
        registry
            .register(Box::new(qdrant_search_requests.clone()))
            .expect("failed to register qdrant_search_requests");
        registry
            .register(Box::new(qdrant_search_duration.clone()))
            .expect("failed to register qdrant_search_duration");
        registry
            .register(Box::new(shm_messages_sent.clone()))
            .expect("failed to register shm_messages_sent");
        registry
            .register(Box::new(shm_messages_received.clone()))
            .expect("failed to register shm_messages_received");
        registry
            .register(Box::new(shm_message_size_bytes.clone()))
            .expect("failed to register shm_message_size_bytes");
        registry
            .register(Box::new(grpc_requests_total.clone()))
            .expect("failed to register grpc_requests_total");
        registry
            .register(Box::new(grpc_request_duration.clone()))
            .expect("failed to register grpc_request_duration");
        registry
            .register(Box::new(grpc_active_connections.clone()))
            .expect("failed to register grpc_active_connections");
        registry
            .register(Box::new(exchange_message_total.clone()))
            .expect("failed to register exchange_message_total");
        registry
            .register(Box::new(exchange_message_parse_errors.clone()))
            .expect("failed to register exchange_message_parse_errors");
        registry
            .register(Box::new(exchange_health_status.clone()))
            .expect("failed to register exchange_health_status");
        registry
            .register(Box::new(exchange_latency_seconds.clone()))
            .expect("failed to register exchange_latency_seconds");
        registry
            .register(Box::new(circuit_breaker_state.clone()))
            .expect("failed to register circuit_breaker_state");
        registry
            .register(Box::new(circuit_breaker_trips.clone()))
            .expect("failed to register circuit_breaker_trips");
        registry
            .register(Box::new(memory_usage_bytes.clone()))
            .expect("failed to register memory_usage_bytes");
        registry
            .register(Box::new(cpu_usage_percent.clone()))
            .expect("failed to register cpu_usage_percent");
        registry
            .register(Box::new(active_tasks.clone()))
            .expect("failed to register active_tasks");

        Self {
            registry,
            system_health_score,
            system_status,
            system_uptime_seconds,
            component_health,
            component_response_time_ms,
            component_probe_total,
            component_probe_failures,
            forward_active_engines,
            forward_orders_submitted,
            forward_orders_filled,
            forward_orders_rejected,
            backward_training_iterations,
            backward_memory_consolidations,
            backward_regime_updates,
            gateway_http_requests,
            gateway_http_request_duration,
            redis_connections_active,
            redis_commands_total,
            redis_command_duration,
            qdrant_vectors_stored,
            qdrant_search_requests,
            qdrant_search_duration,
            shm_messages_sent,
            shm_messages_received,
            shm_message_size_bytes,
            grpc_requests_total,
            grpc_request_duration,
            grpc_active_connections,
            exchange_message_total,
            exchange_message_parse_errors,
            exchange_health_status,
            exchange_latency_seconds,
            circuit_breaker_state,
            circuit_breaker_trips,
            memory_usage_bytes,
            cpu_usage_percent,
            active_tasks,
        }
    }

    /// Update metrics from a health signal
    pub fn update_from_signal(&self, signal: &HealthSignal) {
        // Update system-level metrics
        self.system_health_score.set(signal.health_score());
        self.system_status
            .set(system_status_to_int(signal.system_status));
        self.system_uptime_seconds.set(signal.uptime_seconds as i64);

        // Update component-level metrics
        for component in &signal.components {
            let component_label = component.component_type.to_string();

            // Set component health score
            self.component_health
                .with_label_values(&[&component_label])
                .set(component.status.score());

            // Set response time if available
            if let Some(response_time) = component.response_time_ms {
                self.component_response_time_ms
                    .with_label_values(&[&component_label])
                    .set(response_time as f64);
            }

            // Increment probe counter
            self.component_probe_total
                .with_label_values(&[&component_label, &component.status.to_string()])
                .inc();
        }
    }

    /// Record a component probe failure
    pub fn record_probe_failure(&self, component: ComponentType, reason: &str) {
        let component_str = component.to_string();
        let reason_str = reason.to_string();
        self.component_probe_failures
            .with_label_values(&[&component_str, &reason_str])
            .inc();
    }

    /// Get metrics in Prometheus text format
    pub fn gather(&self) -> Vec<prometheus::proto::MetricFamily> {
        self.registry.gather()
    }
}

impl Default for MetricsRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// CNS Metrics collector - convenience wrapper
pub struct CNSMetrics;

impl CNSMetrics {
    /// Get the global metrics registry
    pub fn registry() -> Arc<MetricsRegistry> {
        METRICS_REGISTRY.clone()
    }

    /// Update metrics from a health signal
    pub fn update(signal: &HealthSignal) {
        METRICS_REGISTRY.update_from_signal(signal);
    }

    /// Record a probe failure
    pub fn record_failure(component: ComponentType, reason: &str) {
        METRICS_REGISTRY.record_probe_failure(component, reason);
    }

    /// Gather all metrics
    pub fn gather() -> Vec<prometheus::proto::MetricFamily> {
        METRICS_REGISTRY.gather()
    }

    /// Encode metrics to Prometheus text format
    pub fn encode_text() -> Result<String, String> {
        use prometheus::Encoder;
        let encoder = prometheus::TextEncoder::new();
        let metric_families = Self::gather();
        let mut buffer = Vec::new();
        encoder
            .encode(&metric_families, &mut buffer)
            .map_err(|e| format!("Failed to encode metrics: {}", e))?;
        String::from_utf8(buffer).map_err(|e| format!("Failed to convert metrics to UTF-8: {}", e))
    }
}

/// Convert SystemStatus to integer for Prometheus
fn system_status_to_int(status: SystemStatus) -> i64 {
    match status {
        SystemStatus::Starting => 0,
        SystemStatus::Healthy => 1,
        SystemStatus::Degraded => 2,
        SystemStatus::Critical => 3,
        SystemStatus::Shutdown => 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signals::ComponentHealth;

    #[test]
    fn test_metrics_registry_creation() {
        let registry = MetricsRegistry::new();
        // Metric families are only included in gather() when they have values.
        // The exact count depends on which metrics have been observed.
        // Just verify we can create a registry and gather returns some metrics.
        let metrics = registry.gather();
        // At minimum we should have the non-vec metrics (gauges, counters without labels)
        assert!(
            metrics.len() >= 10,
            "Expected at least 10 metric families, got {}",
            metrics.len()
        );
    }

    #[test]
    fn test_update_from_signal() {
        let registry = MetricsRegistry::new();
        let components = vec![
            ComponentHealth::healthy(ComponentType::ForwardService).with_response_time(10),
            ComponentHealth::degraded(ComponentType::BackwardService, "slow")
                .with_response_time(500),
        ];
        let signal = HealthSignal::new(components, 3600);

        registry.update_from_signal(&signal);

        // Verify system metrics were updated
        assert_eq!(registry.system_uptime_seconds.get(), 3600);
        assert!(registry.system_health_score.get() > 0.0);
    }

    #[test]
    fn test_encode_text() {
        // First update the global registry with some data to ensure metrics are present
        let components =
            vec![ComponentHealth::healthy(ComponentType::ForwardService).with_response_time(10)];
        let signal = HealthSignal::new(components, 100);
        CNSMetrics::update(&signal);

        let text = CNSMetrics::encode_text().unwrap();
        // Check for system-level metrics that should always be present
        assert!(
            text.contains("janus_cns_system_health_score"),
            "Expected janus_cns_system_health_score in output"
        );
        // After updating with a signal, component health should be present
        assert!(
            text.contains("janus_cns_component_health"),
            "Expected janus_cns_component_health in output after signal update"
        );
    }
}
