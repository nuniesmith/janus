//! Unified Prometheus metrics for all JANUS modules

use prometheus::{
    Counter, CounterVec, Gauge, GaugeVec, HistogramVec, register_counter, register_counter_vec,
    register_gauge, register_gauge_vec, register_histogram_vec,
};
use std::sync::OnceLock;

/// Global metrics instance
static METRICS: OnceLock<JanusMetrics> = OnceLock::new();

/// Get or initialize the global metrics
pub fn metrics() -> &'static JanusMetrics {
    METRICS.get_or_init(JanusMetrics::new)
}

/// Unified metrics for all JANUS modules
pub struct JanusMetrics {
    // =========================================================================
    // Signal Metrics
    // =========================================================================
    /// Total signals generated
    pub signals_generated_total: CounterVec,
    /// Signal generation latency
    pub signal_generation_duration: HistogramVec,
    /// Current signal confidence
    pub signal_confidence: GaugeVec,

    // =========================================================================
    // Module Metrics
    // =========================================================================
    /// Module health (1 = healthy, 0 = unhealthy)
    pub module_health: GaugeVec,
    /// Module uptime in seconds
    pub module_uptime_seconds: GaugeVec,

    // =========================================================================
    // API Metrics
    // =========================================================================
    /// HTTP requests total
    pub http_requests_total: CounterVec,
    /// HTTP request duration
    pub http_request_duration: HistogramVec,
    /// HTTP requests in flight
    pub http_requests_in_flight: Gauge,

    // =========================================================================
    // Database Metrics
    // =========================================================================
    /// Database query count
    pub db_queries_total: CounterVec,
    /// Database query duration
    pub db_query_duration: HistogramVec,
    /// Database connection pool size
    pub db_pool_size: Gauge,

    // =========================================================================
    // Redis Metrics
    // =========================================================================
    /// Redis operations total
    pub redis_operations_total: CounterVec,
    /// Redis operation duration
    pub redis_operation_duration: HistogramVec,
    /// Redis connection status
    pub redis_connected: Gauge,

    // =========================================================================
    // WebSocket Metrics
    // =========================================================================
    /// Active WebSocket connections
    pub websocket_connections: Gauge,
    /// WebSocket messages sent
    pub websocket_messages_sent: CounterVec,

    // =========================================================================
    // Supervisor Metrics
    // =========================================================================
    /// Total number of service restarts across all supervised services.
    /// Maps to: `janus_supervisor_restarts_total`
    pub supervisor_restarts_total: Counter,
    /// Number of services currently in a non-terminal phase.
    /// Maps to: `janus_supervisor_active_services`
    pub supervisor_active_services: Gauge,
    /// Total number of services ever spawned (including initial + restarts).
    /// Maps to: `janus_supervisor_spawned_total`
    pub supervisor_spawned_total: Counter,
    /// Total number of services that have terminated.
    /// Maps to: `janus_supervisor_terminated_total`
    pub supervisor_terminated_total: Counter,
    /// Total number of circuit breaker trips.
    /// Maps to: `janus_supervisor_circuit_breaker_trips_total`
    pub supervisor_circuit_breaker_trips: Counter,
    /// Per-service uptime histogram (seconds).
    /// Maps to: `janus_supervisor_uptime_seconds`
    pub supervisor_uptime_seconds: HistogramVec,
}

impl JanusMetrics {
    /// Create new metrics instance
    pub fn new() -> Self {
        Self {
            // Signal metrics
            signals_generated_total: register_counter_vec!(
                "janus_signals_generated_total",
                "Total number of signals generated",
                &["module", "signal_type", "symbol"]
            )
            .unwrap(),

            signal_generation_duration: register_histogram_vec!(
                "janus_signal_generation_duration_seconds",
                "Time spent generating signals",
                &["module"]
            )
            .unwrap(),

            signal_confidence: register_gauge_vec!(
                "janus_signal_confidence",
                "Confidence of the latest signal",
                &["symbol", "signal_type"]
            )
            .unwrap(),

            // Module metrics
            module_health: register_gauge_vec!(
                "janus_module_health",
                "Module health status (1=healthy, 0=unhealthy)",
                &["module"]
            )
            .unwrap(),

            module_uptime_seconds: register_gauge_vec!(
                "janus_module_uptime_seconds",
                "Module uptime in seconds",
                &["module"]
            )
            .unwrap(),

            // API metrics
            http_requests_total: register_counter_vec!(
                "janus_http_requests_total",
                "Total HTTP requests",
                &["method", "path", "status"]
            )
            .unwrap(),

            http_request_duration: register_histogram_vec!(
                "janus_http_request_duration_seconds",
                "HTTP request duration",
                &["method", "path"]
            )
            .unwrap(),

            http_requests_in_flight: register_gauge!(
                "janus_http_requests_in_flight",
                "Number of HTTP requests currently being processed"
            )
            .unwrap(),

            // Database metrics
            db_queries_total: register_counter_vec!(
                "janus_db_queries_total",
                "Total database queries",
                &["operation", "table"]
            )
            .unwrap(),

            db_query_duration: register_histogram_vec!(
                "janus_db_query_duration_seconds",
                "Database query duration",
                &["operation"]
            )
            .unwrap(),

            db_pool_size: register_gauge!("janus_db_pool_size", "Database connection pool size")
                .unwrap(),

            // Redis metrics
            redis_operations_total: register_counter_vec!(
                "janus_redis_operations_total",
                "Total Redis operations",
                &["operation"]
            )
            .unwrap(),

            redis_operation_duration: register_histogram_vec!(
                "janus_redis_operation_duration_seconds",
                "Redis operation duration",
                &["operation"]
            )
            .unwrap(),

            redis_connected: register_gauge!(
                "janus_redis_connected",
                "Redis connection status (1=connected, 0=disconnected)"
            )
            .unwrap(),

            // WebSocket metrics
            websocket_connections: register_gauge!(
                "janus_websocket_connections",
                "Number of active WebSocket connections"
            )
            .unwrap(),

            websocket_messages_sent: register_counter_vec!(
                "janus_websocket_messages_sent_total",
                "Total WebSocket messages sent",
                &["message_type"]
            )
            .unwrap(),

            // Supervisor metrics
            supervisor_restarts_total: register_counter!(
                "janus_supervisor_restarts_total",
                "Total number of service restarts across all supervised services"
            )
            .unwrap(),

            supervisor_active_services: register_gauge!(
                "janus_supervisor_active_services",
                "Number of services currently in a non-terminal phase"
            )
            .unwrap(),

            supervisor_spawned_total: register_counter!(
                "janus_supervisor_spawned_total",
                "Total number of services ever spawned"
            )
            .unwrap(),

            supervisor_terminated_total: register_counter!(
                "janus_supervisor_terminated_total",
                "Total number of services that have terminated"
            )
            .unwrap(),

            supervisor_circuit_breaker_trips: register_counter!(
                "janus_supervisor_circuit_breaker_trips_total",
                "Total number of circuit breaker trips"
            )
            .unwrap(),

            supervisor_uptime_seconds: register_histogram_vec!(
                "janus_supervisor_uptime_seconds",
                "Per-service cumulative uptime in seconds",
                &["service"]
            )
            .unwrap(),
        }
    }

    /// Record a signal generation
    pub fn record_signal(&self, module: &str, signal_type: &str, symbol: &str, confidence: f64) {
        self.signals_generated_total
            .with_label_values(&[module, signal_type, symbol])
            .inc();
        self.signal_confidence
            .with_label_values(&[symbol, signal_type])
            .set(confidence);
    }

    /// Record module health
    pub fn record_module_health(&self, module: &str, healthy: bool, uptime_seconds: f64) {
        self.module_health
            .with_label_values(&[module])
            .set(if healthy { 1.0 } else { 0.0 });
        self.module_uptime_seconds
            .with_label_values(&[module])
            .set(uptime_seconds);
    }

    /// Record HTTP request
    pub fn record_http_request(&self, method: &str, path: &str, status: u16, duration_secs: f64) {
        self.http_requests_total
            .with_label_values(&[method, path, &status.to_string()])
            .inc();
        self.http_request_duration
            .with_label_values(&[method, path])
            .observe(duration_secs);
    }

    // =========================================================================
    // Supervisor metric helpers
    //
    // NOTE: The `supervisor_active_services` **gauge** is managed
    // authoritatively by `SupervisorMetrics` (in `supervisor/mod.rs`)
    // which uses atomic integers as the single source of truth and
    // calls `supervisor_active_services.set(value)` directly.
    //
    // The helpers below intentionally do NOT touch the gauge so that
    // there is exactly one code path responsible for it, eliminating
    // the TOCTOU race that existed when both `SupervisorMetrics` and
    // these helpers independently incremented / decremented the gauge.
    // =========================================================================

    /// Record a service spawn event (counter only — the `active_services`
    /// gauge is managed by `SupervisorMetrics`).
    pub fn record_supervisor_spawn(&self) {
        self.supervisor_spawned_total.inc();
    }

    /// Record a service restart event.
    pub fn record_supervisor_restart(&self) {
        self.supervisor_restarts_total.inc();
    }

    /// Record a service termination event (counter only — the
    /// `active_services` gauge is managed by `SupervisorMetrics`).
    pub fn record_supervisor_termination(&self) {
        self.supervisor_terminated_total.inc();
    }

    /// Record a circuit breaker trip event.
    pub fn record_supervisor_circuit_breaker_trip(&self) {
        self.supervisor_circuit_breaker_trips.inc();
    }

    /// Record a service's cumulative uptime when it terminates.
    pub fn record_supervisor_service_uptime(&self, service: &str, uptime_secs: f64) {
        self.supervisor_uptime_seconds
            .with_label_values(&[service])
            .observe(uptime_secs);
    }

    /// Get all metrics as text (for /metrics endpoint)
    pub fn encode(&self) -> String {
        use prometheus::Encoder;
        let encoder = prometheus::TextEncoder::new();
        let metric_families = prometheus::gather();
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer).unwrap();
        String::from_utf8(buffer).unwrap()
    }
}

impl Default for JanusMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// All tests must use the global `metrics()` singleton because
    /// Prometheus metric names are globally registered — calling
    /// `JanusMetrics::new()` more than once panics on duplicate
    /// registration.  Since tests run in the same process (and the
    /// supervisor code also calls `metrics()`), every test must go
    /// through the singleton.

    #[test]
    fn test_metrics_creation() {
        let m = metrics();
        // Just verify the singleton is usable and helpers don't panic.
        m.record_signal("forward", "buy", "BTCUSD", 0.85);
        m.record_module_health("forward", true, 100.0);
    }

    #[test]
    fn test_supervisor_metrics_helpers() {
        let m = metrics();

        // Because tests run in parallel and share the global Prometheus
        // registry, we cannot assert exact counter deltas (other
        // supervisor tests bump the same counters concurrently).
        // Instead we verify that:
        //   1. The helper methods don't panic
        //   2. Monotonic counters only increase

        let spawned_before = m.supervisor_spawned_total.get();
        m.record_supervisor_spawn();
        assert!(m.supervisor_spawned_total.get() > spawned_before);

        let restarts_before = m.supervisor_restarts_total.get();
        m.record_supervisor_restart();
        assert!(m.supervisor_restarts_total.get() > restarts_before);

        let terminated_before = m.supervisor_terminated_total.get();
        m.record_supervisor_termination();
        assert!(m.supervisor_terminated_total.get() > terminated_before);

        m.record_supervisor_service_uptime("test-svc", 42.5);

        let trips_before = m.supervisor_circuit_breaker_trips.get();
        m.record_supervisor_circuit_breaker_trip();
        assert!(m.supervisor_circuit_breaker_trips.get() > trips_before);
    }

    #[test]
    fn test_supervisor_active_services_gauge_is_settable() {
        let m = metrics();

        // The `active_services` gauge is now managed exclusively by
        // `SupervisorMetrics` via `set()`.  Verify that the gauge
        // can be set to an arbitrary value and read back correctly.
        m.supervisor_active_services.set(5.0);
        assert_eq!(m.supervisor_active_services.get(), 5.0);

        m.supervisor_active_services.set(0.0);
        assert_eq!(m.supervisor_active_services.get(), 0.0);

        // `record_supervisor_spawn` and `record_supervisor_termination`
        // no longer touch the gauge — verify that.
        m.supervisor_active_services.set(3.0);
        m.record_supervisor_spawn();
        assert_eq!(
            m.supervisor_active_services.get(),
            3.0,
            "record_supervisor_spawn should not modify the gauge"
        );
        m.record_supervisor_termination();
        assert_eq!(
            m.supervisor_active_services.get(),
            3.0,
            "record_supervisor_termination should not modify the gauge"
        );

        // Clean up: reset gauge so other tests aren't affected.
        m.supervisor_active_services.set(0.0);
    }
}
