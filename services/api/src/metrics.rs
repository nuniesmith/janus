//! Prometheus metrics for the JANUS Rust Gateway.
//!
//! This module provides metrics collection and a `/metrics` endpoint
//! for Prometheus scraping.

use axum::{Router, extract::State, http::StatusCode, response::IntoResponse, routing::get};
use prometheus::{
    Counter, CounterVec, Encoder, Gauge, HistogramOpts, HistogramVec, Opts, Registry, TextEncoder,
};
use std::sync::Arc;
use std::time::Instant;
use tracing::error;

use crate::state::AppState;

/// Gateway metrics container.
///
/// Holds all Prometheus metrics for the gateway service.
pub struct GatewayMetrics {
    /// Prometheus registry
    registry: Registry,

    /// Total HTTP requests received
    pub http_requests_total: CounterVec,

    /// HTTP request duration in seconds
    pub http_request_duration_seconds: HistogramVec,

    /// Currently active HTTP requests
    pub http_requests_in_flight: Gauge,

    /// Total signals dispatched via Redis
    pub signals_dispatched_total: CounterVec,

    /// Signal dispatch errors
    pub signal_dispatch_errors_total: Counter,

    /// Redis connection status (1 = connected, 0 = disconnected)
    pub redis_connected: Gauge,

    /// gRPC client connection status (1 = connected, 0 = disconnected)
    pub grpc_connected: Gauge,

    /// Heartbeats sent to forward service
    pub heartbeats_sent_total: Counter,

    /// Heartbeat errors
    pub heartbeat_errors_total: Counter,

    /// Gateway uptime in seconds
    pub uptime_seconds: Gauge,

    /// Setup wizard completions
    pub setup_completions: Counter,

    /// Setup wizard resets (used in debug builds via reset_setup endpoint)
    #[allow(dead_code)]
    pub setup_resets: Counter,

    /// Start time for uptime calculation
    pub start_time: Instant,
}

impl GatewayMetrics {
    /// Create a new metrics instance with all metrics registered.
    pub fn new() -> Result<Self, prometheus::Error> {
        let registry = Registry::new();

        // HTTP metrics
        let http_requests_total = CounterVec::new(
            Opts::new(
                "http_requests_total",
                "Total number of HTTP requests received",
            )
            .namespace("janus_gateway"),
            &["method", "path", "status"],
        )?;
        registry.register(Box::new(http_requests_total.clone()))?;

        let http_request_duration_seconds = HistogramVec::new(
            HistogramOpts::new(
                "http_request_duration_seconds",
                "HTTP request duration in seconds",
            )
            .namespace("janus_gateway")
            .buckets(vec![
                0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
            ]),
            &["method", "path"],
        )?;
        registry.register(Box::new(http_request_duration_seconds.clone()))?;

        let http_requests_in_flight = Gauge::new(
            "janus_gateway_http_requests_in_flight",
            "Number of HTTP requests currently being processed",
        )?;
        registry.register(Box::new(http_requests_in_flight.clone()))?;

        // Signal metrics
        let signals_dispatched_total = CounterVec::new(
            Opts::new(
                "signals_dispatched_total",
                "Total number of signals dispatched",
            )
            .namespace("janus_gateway"),
            &["symbol", "side"],
        )?;
        registry.register(Box::new(signals_dispatched_total.clone()))?;

        let signal_dispatch_errors_total = Counter::new(
            "janus_gateway_signal_dispatch_errors_total",
            "Total number of signal dispatch errors",
        )?;
        registry.register(Box::new(signal_dispatch_errors_total.clone()))?;

        // Connection metrics
        let redis_connected = Gauge::new(
            "janus_gateway_redis_connected",
            "Redis connection status (1 = connected, 0 = disconnected)",
        )?;
        registry.register(Box::new(redis_connected.clone()))?;

        let grpc_connected = Gauge::new(
            "janus_gateway_grpc_connected",
            "gRPC client connection status (1 = connected, 0 = disconnected)",
        )?;
        registry.register(Box::new(grpc_connected.clone()))?;

        // Heartbeat metrics
        let heartbeats_sent_total = Counter::new(
            "janus_gateway_heartbeats_sent_total",
            "Total number of heartbeats sent to forward service",
        )?;
        registry.register(Box::new(heartbeats_sent_total.clone()))?;

        let heartbeat_errors_total = Counter::new(
            "janus_gateway_heartbeat_errors_total",
            "Total number of heartbeat errors",
        )?;
        registry.register(Box::new(heartbeat_errors_total.clone()))?;

        // Uptime metric
        let uptime_seconds =
            Gauge::new("janus_gateway_uptime_seconds", "Gateway uptime in seconds")?;
        registry.register(Box::new(uptime_seconds.clone()))?;

        // Setup metrics
        let setup_completions = Counter::new(
            "janus_gateway_setup_completions_total",
            "Total number of successful setup wizard completions",
        )?;
        registry.register(Box::new(setup_completions.clone()))?;

        let setup_resets = Counter::new(
            "janus_gateway_setup_resets_total",
            "Total number of setup wizard resets",
        )?;
        registry.register(Box::new(setup_resets.clone()))?;

        Ok(Self {
            registry,
            http_requests_total,
            http_request_duration_seconds,
            http_requests_in_flight,
            signals_dispatched_total,
            signal_dispatch_errors_total,
            redis_connected,
            grpc_connected,
            heartbeats_sent_total,
            heartbeat_errors_total,
            uptime_seconds,
            setup_completions,
            setup_resets,
            start_time: Instant::now(),
        })
    }

    /// Record an HTTP request.
    pub fn record_request(&self, method: &str, path: &str, status: u16, duration_secs: f64) {
        self.http_requests_total
            .with_label_values(&[method, path, &status.to_string()])
            .inc();
        self.http_request_duration_seconds
            .with_label_values(&[method, path])
            .observe(duration_secs);
    }

    /// Record a signal dispatch.
    #[allow(dead_code)]
    pub fn record_signal_dispatch(&self, symbol: &str, side: &str) {
        self.signals_dispatched_total
            .with_label_values(&[symbol, side])
            .inc();
    }

    /// Record a signal dispatch error.
    #[allow(dead_code)]
    pub fn record_signal_error(&self) {
        self.signal_dispatch_errors_total.inc();
    }

    /// Record a heartbeat sent.
    pub fn record_heartbeat(&self) {
        self.heartbeats_sent_total.inc();
    }

    /// Record a heartbeat error.
    pub fn record_heartbeat_error(&self) {
        self.heartbeat_errors_total.inc();
    }

    /// Update connection status metrics.
    pub fn update_connection_status(&self, redis_connected: bool, grpc_connected: bool) {
        self.redis_connected
            .set(if redis_connected { 1.0 } else { 0.0 });
        self.grpc_connected
            .set(if grpc_connected { 1.0 } else { 0.0 });
    }

    /// Update uptime metric.
    pub fn update_uptime(&self) {
        self.uptime_seconds
            .set(self.start_time.elapsed().as_secs_f64());
    }

    /// Compute the average HTTP request latency in milliseconds from the
    /// `http_request_duration_seconds` histogram.
    ///
    /// Returns `None` if no requests have been recorded yet.
    pub fn avg_request_latency_ms(&self) -> Option<f64> {
        let metric_families = self.registry.gather();
        for mf in &metric_families {
            if mf.name() == "http_request_duration_seconds" {
                let mut total_sum: f64 = 0.0;
                let mut total_count: u64 = 0;
                for m in mf.get_metric() {
                    let h = m.get_histogram();
                    total_sum += h.get_sample_sum();
                    total_count += h.get_sample_count();
                }
                if total_count > 0 {
                    return Some((total_sum / total_count as f64) * 1000.0);
                }
            }
        }
        None
    }

    /// Encode metrics in Prometheus text format.
    pub fn encode(&self) -> Result<String, prometheus::Error> {
        // Update uptime before encoding
        self.update_uptime();

        let encoder = TextEncoder::new();
        let metric_families = self.registry.gather();
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer)?;
        String::from_utf8(buffer)
            .map_err(|e| prometheus::Error::Msg(format!("UTF-8 encoding error: {}", e)))
    }
}

impl Default for GatewayMetrics {
    fn default() -> Self {
        Self::new().expect("Failed to create default metrics")
    }
}

/// Metrics endpoint handler.
///
/// Returns metrics in Prometheus text exposition format.
pub async fn metrics_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    // Update connection status
    let redis_connected = state.signal_dispatcher.is_connected().await;
    let grpc_connected = state.janus_client.read().await.is_some();
    state
        .metrics
        .update_connection_status(redis_connected, grpc_connected);

    match state.metrics.encode() {
        Ok(metrics) => (
            StatusCode::OK,
            [(
                axum::http::header::CONTENT_TYPE,
                "text/plain; version=0.0.4; charset=utf-8",
            )],
            metrics,
        )
            .into_response(),
        Err(e) => {
            error!("Failed to encode metrics: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Failed to encode metrics",
            )
                .into_response()
        }
    }
}

/// Create the metrics routes router.
pub fn metrics_routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/metrics", get(metrics_handler))
        .route("/api/v1/metrics", get(metrics_handler))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_creation() {
        let metrics = GatewayMetrics::new();
        assert!(metrics.is_ok());
    }

    #[test]
    fn test_metrics_recording() {
        let metrics = GatewayMetrics::new().unwrap();

        // Record some metrics
        metrics.record_request("GET", "/health", 200, 0.001);
        metrics.record_request("POST", "/api/signals/dispatch", 200, 0.005);
        metrics.record_signal_dispatch("BTCUSD", "Buy");
        metrics.record_heartbeat();

        // Verify they can be encoded
        let encoded = metrics.encode();
        assert!(encoded.is_ok());

        let text = encoded.unwrap();
        assert!(text.contains("http_requests_total"));
        assert!(text.contains("signals_dispatched_total"));
        assert!(text.contains("heartbeats_sent_total"));
    }

    #[test]
    fn test_connection_status() {
        let metrics = GatewayMetrics::new().unwrap();

        metrics.update_connection_status(true, false);

        let encoded = metrics.encode().unwrap();
        assert!(encoded.contains("redis_connected"));
        assert!(encoded.contains("grpc_connected"));
    }

    #[test]
    fn test_uptime_metric() {
        let metrics = GatewayMetrics::new().unwrap();

        // Sleep a tiny bit
        std::thread::sleep(std::time::Duration::from_millis(10));

        metrics.update_uptime();

        let encoded = metrics.encode().unwrap();
        assert!(encoded.contains("uptime_seconds"));
    }

    #[test]
    fn test_histogram_buckets() {
        let metrics = GatewayMetrics::new().unwrap();

        // Record various durations
        metrics.record_request("GET", "/test", 200, 0.001);
        metrics.record_request("GET", "/test", 200, 0.01);
        metrics.record_request("GET", "/test", 200, 0.1);
        metrics.record_request("GET", "/test", 200, 1.0);

        let encoded = metrics.encode().unwrap();
        assert!(encoded.contains("http_request_duration_seconds_bucket"));
    }
}
