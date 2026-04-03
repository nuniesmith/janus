//! # Prometheus Metrics Module
//!
//! Comprehensive metrics collection and export for JANUS service using Prometheus.
//! Tracks signal generation, risk management, ML inference, and system performance.

pub mod brain_metrics;
pub mod prometheus_exporter;
pub mod risk_metrics;
pub mod signal_metrics;

use prometheus::{Gauge, Histogram, HistogramOpts, IntCounter, IntGauge, Opts, Registry};
use std::sync::Arc;

pub use brain_metrics::BrainPipelineMetricsCollector;
pub use prometheus_exporter::PrometheusExporter;
pub use risk_metrics::RiskMetricsCollector;
pub use signal_metrics::SignalMetricsCollector;

/// Main metrics collector for JANUS service
pub struct JanusMetrics {
    registry: Arc<Registry>,
    signal_metrics: SignalMetricsCollector,
    risk_metrics: RiskMetricsCollector,
    system_metrics: SystemMetrics,
    websocket_metrics: WebSocketMetrics,
    brain_metrics: Option<BrainPipelineMetricsCollector>,
}

impl JanusMetrics {
    /// Create new metrics collector
    pub fn new() -> Result<Self, prometheus::Error> {
        let registry = Arc::new(Registry::new());

        let signal_metrics = SignalMetricsCollector::new(Arc::clone(&registry))?;
        let risk_metrics = RiskMetricsCollector::new(Arc::clone(&registry))?;
        let system_metrics = SystemMetrics::new(Arc::clone(&registry))?;
        let websocket_metrics = WebSocketMetrics::new(Arc::clone(&registry))?;

        // Brain metrics are optional — they may fail to register if another
        // collector already owns the metric names (e.g. in tests), so we
        // treat registration failure as non-fatal.
        let brain_metrics = BrainPipelineMetricsCollector::new(Arc::clone(&registry)).ok();

        Ok(Self {
            registry,
            signal_metrics,
            risk_metrics,
            system_metrics,
            websocket_metrics,
            brain_metrics,
        })
    }

    /// Get signal metrics collector
    pub fn signal_metrics(&self) -> &SignalMetricsCollector {
        &self.signal_metrics
    }

    /// Get risk metrics collector
    pub fn risk_metrics(&self) -> &RiskMetricsCollector {
        &self.risk_metrics
    }

    /// Get system metrics collector
    pub fn system_metrics(&self) -> &SystemMetrics {
        &self.system_metrics
    }

    /// Get WebSocket metrics collector
    pub fn websocket_metrics(&self) -> &WebSocketMetrics {
        &self.websocket_metrics
    }

    /// Get brain pipeline metrics collector (if registered).
    pub fn brain_metrics(&self) -> Option<&BrainPipelineMetricsCollector> {
        self.brain_metrics.as_ref()
    }

    /// Get Prometheus registry
    pub fn registry(&self) -> Arc<Registry> {
        Arc::clone(&self.registry)
    }

    /// Gather all metrics in Prometheus format
    pub fn gather(&self) -> Vec<prometheus::proto::MetricFamily> {
        self.registry.gather()
    }
}

impl Default for JanusMetrics {
    fn default() -> Self {
        Self::new().expect("Failed to create metrics collector")
    }
}

/// System-level metrics
pub struct SystemMetrics {
    // Request metrics
    pub http_requests_total: IntCounter,
    pub grpc_requests_total: IntCounter,
    pub http_request_duration: Histogram,
    pub grpc_request_duration: Histogram,

    // Error metrics
    pub errors_total: IntCounter,
    pub error_rate: Gauge,

    // System metrics
    pub uptime_seconds: Gauge,
    pub active_connections: IntGauge,
    pub memory_usage_bytes: Gauge,
}

impl SystemMetrics {
    fn new(registry: Arc<Registry>) -> Result<Self, prometheus::Error> {
        let http_requests_total = IntCounter::with_opts(Opts::new(
            "janus_http_requests_total",
            "Total number of HTTP requests",
        ))?;
        registry.register(Box::new(http_requests_total.clone()))?;

        let grpc_requests_total = IntCounter::with_opts(Opts::new(
            "janus_grpc_requests_total",
            "Total number of gRPC requests",
        ))?;
        registry.register(Box::new(grpc_requests_total.clone()))?;

        let http_request_duration = Histogram::with_opts(HistogramOpts::new(
            "janus_http_request_duration_seconds",
            "HTTP request duration in seconds",
        ))?;
        registry.register(Box::new(http_request_duration.clone()))?;

        let grpc_request_duration = Histogram::with_opts(HistogramOpts::new(
            "janus_grpc_request_duration_seconds",
            "gRPC request duration in seconds",
        ))?;
        registry.register(Box::new(grpc_request_duration.clone()))?;

        let errors_total =
            IntCounter::with_opts(Opts::new("janus_errors_total", "Total number of errors"))?;
        registry.register(Box::new(errors_total.clone()))?;

        let error_rate = Gauge::with_opts(Opts::new(
            "janus_error_rate",
            "Error rate (errors per second)",
        ))?;
        registry.register(Box::new(error_rate.clone()))?;

        let uptime_seconds = Gauge::with_opts(Opts::new(
            "janus_uptime_seconds",
            "Service uptime in seconds",
        ))?;
        registry.register(Box::new(uptime_seconds.clone()))?;

        let active_connections = IntGauge::with_opts(Opts::new(
            "janus_active_connections",
            "Number of active connections",
        ))?;
        registry.register(Box::new(active_connections.clone()))?;

        let memory_usage_bytes = Gauge::with_opts(Opts::new(
            "janus_memory_usage_bytes",
            "Memory usage in bytes",
        ))?;
        registry.register(Box::new(memory_usage_bytes.clone()))?;

        Ok(Self {
            http_requests_total,
            grpc_requests_total,
            http_request_duration,
            grpc_request_duration,
            errors_total,
            error_rate,
            uptime_seconds,
            active_connections,
            memory_usage_bytes,
        })
    }
}

/// WebSocket-specific metrics
pub struct WebSocketMetrics {
    // Connection metrics
    pub ws_connections_total: IntCounter,
    pub ws_connections_active: IntGauge,
    pub ws_messages_sent_total: IntCounter,
    pub ws_messages_received_total: IntCounter,
    pub ws_errors_total: IntCounter,

    // Broadcasting metrics
    pub signal_broadcast_count: IntCounter,
    pub risk_alert_count: IntCounter,

    // Performance metrics
    pub ws_message_latency: Histogram,
    pub ws_broadcast_duration: Histogram,
    pub ws_subscription_updates: IntCounter,

    // Data service metrics
    pub data_service_connection_status: IntGauge,
    pub data_service_reconnections_total: IntCounter,
    pub data_service_messages_received: IntCounter,
    pub data_service_lag_seconds: Histogram,
}

impl WebSocketMetrics {
    fn new(registry: Arc<Registry>) -> Result<Self, prometheus::Error> {
        let ws_connections_total = IntCounter::with_opts(Opts::new(
            "janus_ws_connections_total",
            "Total number of WebSocket connections",
        ))?;
        registry.register(Box::new(ws_connections_total.clone()))?;

        let ws_connections_active = IntGauge::with_opts(Opts::new(
            "janus_ws_connections_active",
            "Number of active WebSocket connections",
        ))?;
        registry.register(Box::new(ws_connections_active.clone()))?;

        let ws_messages_sent_total = IntCounter::with_opts(Opts::new(
            "janus_ws_messages_sent_total",
            "Total number of WebSocket messages sent",
        ))?;
        registry.register(Box::new(ws_messages_sent_total.clone()))?;

        let ws_messages_received_total = IntCounter::with_opts(Opts::new(
            "janus_ws_messages_received_total",
            "Total number of WebSocket messages received",
        ))?;
        registry.register(Box::new(ws_messages_received_total.clone()))?;

        let ws_errors_total = IntCounter::with_opts(Opts::new(
            "janus_ws_errors_total",
            "Total number of WebSocket errors",
        ))?;
        registry.register(Box::new(ws_errors_total.clone()))?;

        let signal_broadcast_count = IntCounter::with_opts(Opts::new(
            "janus_signal_broadcast_count",
            "Total number of signals broadcast via WebSocket",
        ))?;
        registry.register(Box::new(signal_broadcast_count.clone()))?;

        let risk_alert_count = IntCounter::with_opts(Opts::new(
            "janus_risk_alert_count",
            "Total number of risk alerts broadcast",
        ))?;
        registry.register(Box::new(risk_alert_count.clone()))?;

        let ws_message_latency = Histogram::with_opts(HistogramOpts::new(
            "janus_ws_message_latency_seconds",
            "WebSocket message latency in seconds",
        ))?;
        registry.register(Box::new(ws_message_latency.clone()))?;

        let ws_broadcast_duration = Histogram::with_opts(HistogramOpts::new(
            "janus_ws_broadcast_duration_seconds",
            "WebSocket broadcast duration in seconds",
        ))?;
        registry.register(Box::new(ws_broadcast_duration.clone()))?;

        let ws_subscription_updates = IntCounter::with_opts(Opts::new(
            "janus_ws_subscription_updates_total",
            "Total number of subscription updates",
        ))?;
        registry.register(Box::new(ws_subscription_updates.clone()))?;

        let data_service_connection_status = IntGauge::with_opts(Opts::new(
            "janus_data_service_connection_status",
            "Data service connection status (1=connected, 0=disconnected)",
        ))?;
        registry.register(Box::new(data_service_connection_status.clone()))?;

        let data_service_reconnections_total = IntCounter::with_opts(Opts::new(
            "janus_data_service_reconnections_total",
            "Total number of data service reconnections",
        ))?;
        registry.register(Box::new(data_service_reconnections_total.clone()))?;

        let data_service_messages_received = IntCounter::with_opts(Opts::new(
            "janus_data_service_messages_received_total",
            "Total number of messages received from data service",
        ))?;
        registry.register(Box::new(data_service_messages_received.clone()))?;

        let data_service_lag_seconds = Histogram::with_opts(HistogramOpts::new(
            "janus_data_service_lag_seconds",
            "Data service message lag in seconds",
        ))?;
        registry.register(Box::new(data_service_lag_seconds.clone()))?;

        Ok(Self {
            ws_connections_total,
            ws_connections_active,
            ws_messages_sent_total,
            ws_messages_received_total,
            ws_errors_total,
            signal_broadcast_count,
            risk_alert_count,
            ws_message_latency,
            ws_broadcast_duration,
            ws_subscription_updates,
            data_service_connection_status,
            data_service_reconnections_total,
            data_service_messages_received,
            data_service_lag_seconds,
        })
    }
}

impl SystemMetrics {
    /// Record HTTP request
    pub fn record_http_request(&self, duration_secs: f64) {
        self.http_requests_total.inc();
        self.http_request_duration.observe(duration_secs);
    }

    /// Record gRPC request
    pub fn record_grpc_request(&self, duration_secs: f64) {
        self.grpc_requests_total.inc();
        self.grpc_request_duration.observe(duration_secs);
    }

    /// Record error
    pub fn record_error(&self) {
        self.errors_total.inc();
    }

    /// Update uptime
    pub fn update_uptime(&self, seconds: f64) {
        self.uptime_seconds.set(seconds);
    }

    /// Set active connections
    pub fn set_active_connections(&self, count: i64) {
        self.active_connections.set(count);
    }

    /// Update memory usage
    pub fn update_memory_usage(&self, bytes: f64) {
        self.memory_usage_bytes.set(bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_creation() {
        let metrics = JanusMetrics::new();
        assert!(metrics.is_ok());
    }

    #[test]
    fn test_metrics_default() {
        let metrics = JanusMetrics::default();
        assert!(!metrics.registry.gather().is_empty());
    }

    #[test]
    fn test_system_metrics_recording() {
        let metrics = JanusMetrics::default();

        // Record some metrics
        metrics.system_metrics().record_http_request(0.1);
        metrics.system_metrics().record_grpc_request(0.05);
        metrics.system_metrics().record_error();
        metrics.system_metrics().update_uptime(100.0);
        metrics.system_metrics().set_active_connections(5);

        // Gather metrics
        let gathered = metrics.gather();
        assert!(!gathered.is_empty());
    }

    #[test]
    fn test_multiple_http_requests() {
        let metrics = JanusMetrics::default();

        for _ in 0..10 {
            metrics.system_metrics().record_http_request(0.1);
        }

        // Verify counter incremented
        assert_eq!(metrics.system_metrics().http_requests_total.get(), 10);
    }
}
