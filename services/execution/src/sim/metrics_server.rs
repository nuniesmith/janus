//! Standalone Metrics HTTP Server for Simulation Components
//!
//! Provides a lightweight HTTP server that exposes Prometheus metrics for
//! DataRecorder, LiveFeedBridge, and ReplayEngine components.
//!
//! ## Usage
//!
//! ### Basic Server
//!
//! ```rust,ignore
//! use janus_execution::sim::metrics_server::{SimMetricsServer, SimMetricsServerConfig};
//! use janus_execution::sim::{DataRecorder, LiveFeedBridge};
//!
//! // Create server with default config (port 9090)
//! let server = SimMetricsServer::new(SimMetricsServerConfig::default());
//!
//! // Add components to monitor
//! let recorder = Arc::new(DataRecorder::new(config).await?);
//! server.set_recorder(recorder.clone());
//!
//! // Start serving metrics
//! let handle = server.start().await?;
//!
//! // ... run your application ...
//!
//! // Shutdown
//! handle.shutdown().await;
//! ```
//!
//! ### Using the Builder
//!
//! ```rust,ignore
//! let server = SimMetricsServer::builder()
//!     .port(9091)
//!     .update_interval_ms(500)
//!     .recorder(recorder)
//!     .bridge(bridge)
//!     .build();
//!
//! server.start().await?;
//! ```
//!
//! ## Endpoints
//!
//! - `GET /metrics` - Prometheus-format metrics
//! - `GET /health` - Health check (returns "OK")
//! - `GET /` - Root endpoint (same as health)

use super::data_recorder::DataRecorder;
use super::live_feed_bridge::LiveFeedBridge;
use super::metrics::{global_sim_metrics, sim_prometheus_metrics};
use super::replay::ReplayEngine;
use axum::{Router, routing::get};
use parking_lot::RwLock;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use tracing::{debug, error, info, warn};

/// Errors that can occur with the metrics server
#[derive(Debug, Error)]
pub enum MetricsServerError {
    #[error("Failed to bind to address {0}: {1}")]
    BindFailed(SocketAddr, std::io::Error),

    #[error("Server error: {0}")]
    ServerError(String),

    #[error("Server not running")]
    NotRunning,

    #[error("Server already running")]
    AlreadyRunning,
}

/// Configuration for the sim metrics server
#[derive(Debug, Clone)]
pub struct SimMetricsServerConfig {
    /// Port to listen on
    pub port: u16,
    /// Host to bind to
    pub host: String,
    /// Interval for updating metrics from components (milliseconds)
    pub update_interval_ms: u64,
    /// Whether to try alternative ports if the primary port is in use
    pub try_alternative_ports: bool,
    /// Instance name for metric labels
    pub instance_name: String,
}

impl Default for SimMetricsServerConfig {
    fn default() -> Self {
        Self {
            port: 9090,
            host: "0.0.0.0".to_string(),
            update_interval_ms: 1000,
            try_alternative_ports: true,
            instance_name: "sim".to_string(),
        }
    }
}

impl SimMetricsServerConfig {
    /// Create a new configuration with the specified port
    pub fn new(port: u16) -> Self {
        Self {
            port,
            ..Default::default()
        }
    }

    /// Set the host to bind to
    pub fn with_host(mut self, host: impl Into<String>) -> Self {
        self.host = host.into();
        self
    }

    /// Set the update interval
    pub fn with_update_interval(mut self, interval: Duration) -> Self {
        self.update_interval_ms = interval.as_millis() as u64;
        self
    }

    /// Set whether to try alternative ports
    pub fn with_alternative_ports(mut self, enabled: bool) -> Self {
        self.try_alternative_ports = enabled;
        self
    }

    /// Set the instance name for metric labels
    pub fn with_instance_name(mut self, name: impl Into<String>) -> Self {
        self.instance_name = name.into();
        self
    }
}

/// Handle for controlling a running metrics server
pub struct SimMetricsServerHandle {
    /// Shutdown signal sender
    shutdown_tx: Option<oneshot::Sender<()>>,
    /// Server task handle
    server_handle: JoinHandle<()>,
    /// Collector task handle
    collector_handle: Option<JoinHandle<()>>,
    /// The address the server is bound to
    bound_addr: SocketAddr,
}

impl SimMetricsServerHandle {
    /// Get the address the server is bound to
    pub fn address(&self) -> SocketAddr {
        self.bound_addr
    }

    /// Get the port the server is bound to
    pub fn port(&self) -> u16 {
        self.bound_addr.port()
    }

    /// Shutdown the server gracefully
    pub async fn shutdown(mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }

        // Abort tasks
        self.server_handle.abort();
        if let Some(handle) = self.collector_handle.take() {
            handle.abort();
        }

        info!("Metrics server shutdown complete");
    }

    /// Check if the server is still running
    pub fn is_running(&self) -> bool {
        !self.server_handle.is_finished()
    }
}

/// Shared state for the metrics server
struct ServerState {
    recorder: Option<Arc<DataRecorder>>,
    bridge: Option<Arc<LiveFeedBridge>>,
    #[allow(dead_code)]
    replay: Option<Arc<ReplayEngine>>,
}

/// Standalone HTTP server for sim metrics
pub struct SimMetricsServer {
    config: SimMetricsServerConfig,
    state: Arc<RwLock<ServerState>>,
}

impl SimMetricsServer {
    /// Create a new metrics server with the given configuration
    pub fn new(config: SimMetricsServerConfig) -> Self {
        Self {
            config,
            state: Arc::new(RwLock::new(ServerState {
                recorder: None,
                bridge: None,
                replay: None,
            })),
        }
    }

    /// Create a builder for the metrics server
    pub fn builder() -> SimMetricsServerBuilder {
        SimMetricsServerBuilder::new()
    }

    /// Set the data recorder to monitor
    pub fn set_recorder(&self, recorder: Arc<DataRecorder>) {
        self.state.write().recorder = Some(recorder);
    }

    /// Set the live feed bridge to monitor
    pub fn set_bridge(&self, bridge: Arc<LiveFeedBridge>) {
        self.state.write().bridge = Some(bridge);
    }

    /// Set the replay engine to monitor
    pub fn set_replay(&self, replay: Arc<ReplayEngine>) {
        self.state.write().replay = Some(replay);
    }

    /// Start the metrics server
    ///
    /// Returns a handle that can be used to control the server
    pub async fn start(self) -> Result<SimMetricsServerHandle, MetricsServerError> {
        let addr = self.find_available_port().await?;

        info!(
            "Starting sim metrics server on http://{}:{}/metrics",
            addr.ip(),
            addr.port()
        );

        // Create the router
        let app = Router::new()
            .route("/metrics", get(metrics_handler))
            .route("/health", get(health_handler))
            .route("/", get(health_handler));

        // Bind the listener
        let listener = TcpListener::bind(addr)
            .await
            .map_err(|e| MetricsServerError::BindFailed(addr, e))?;

        let bound_addr = listener
            .local_addr()
            .map_err(|e| MetricsServerError::ServerError(e.to_string()))?;

        // Create shutdown channel
        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();

        // Start the metrics collector task
        let state_clone = self.state.clone();
        let update_interval = self.config.update_interval_ms;
        let collector_handle = tokio::spawn(async move {
            metrics_collector_task(state_clone, update_interval).await;
        });

        // Start the server task
        let server_handle = tokio::spawn(async move {
            tokio::select! {
                result = axum::serve(listener, app) => {
                    if let Err(e) = result {
                        error!("Metrics server error: {}", e);
                    }
                }
                _ = shutdown_rx => {
                    info!("Metrics server received shutdown signal");
                }
            }
        });

        info!(
            "📊 Sim metrics server listening on http://{}/metrics",
            bound_addr
        );

        Ok(SimMetricsServerHandle {
            shutdown_tx: Some(shutdown_tx),
            server_handle,
            collector_handle: Some(collector_handle),
            bound_addr,
        })
    }

    /// Find an available port, trying alternatives if configured
    async fn find_available_port(&self) -> Result<SocketAddr, MetricsServerError> {
        let base_port = self.config.port;
        let host = &self.config.host;

        let ports_to_try: Vec<u16> = if self.config.try_alternative_ports {
            vec![base_port, base_port + 1, base_port + 2, base_port + 10, 0]
        } else {
            vec![base_port]
        };

        for port in ports_to_try {
            let addr_str = format!("{}:{}", host, port);
            let addr: SocketAddr = match addr_str.parse() {
                Ok(a) => a,
                Err(_) => continue,
            };

            match TcpListener::bind(addr).await {
                Ok(listener) => {
                    let bound_addr = listener.local_addr().unwrap_or(addr);
                    drop(listener); // Release the port so we can bind again later

                    if port != base_port && port != 0 {
                        warn!(
                            "Port {} was in use, will use port {} instead",
                            base_port,
                            bound_addr.port()
                        );
                    }

                    return Ok(bound_addr);
                }
                Err(e) => {
                    debug!("Port {} unavailable: {}", port, e);
                    if !self.config.try_alternative_ports {
                        return Err(MetricsServerError::BindFailed(addr, e));
                    }
                }
            }
        }

        Err(MetricsServerError::ServerError(
            "All ports unavailable".to_string(),
        ))
    }
}

impl Default for SimMetricsServer {
    fn default() -> Self {
        Self::new(SimMetricsServerConfig::default())
    }
}

/// Builder for SimMetricsServer
pub struct SimMetricsServerBuilder {
    config: SimMetricsServerConfig,
    recorder: Option<Arc<DataRecorder>>,
    bridge: Option<Arc<LiveFeedBridge>>,
    replay: Option<Arc<ReplayEngine>>,
}

impl SimMetricsServerBuilder {
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self {
            config: SimMetricsServerConfig::default(),
            recorder: None,
            bridge: None,
            replay: None,
        }
    }

    /// Set the port
    pub fn port(mut self, port: u16) -> Self {
        self.config.port = port;
        self
    }

    /// Set the host
    pub fn host(mut self, host: impl Into<String>) -> Self {
        self.config.host = host.into();
        self
    }

    /// Set the update interval in milliseconds
    pub fn update_interval_ms(mut self, ms: u64) -> Self {
        self.config.update_interval_ms = ms;
        self
    }

    /// Set the update interval
    pub fn update_interval(mut self, interval: Duration) -> Self {
        self.config.update_interval_ms = interval.as_millis() as u64;
        self
    }

    /// Enable or disable trying alternative ports
    pub fn try_alternative_ports(mut self, enabled: bool) -> Self {
        self.config.try_alternative_ports = enabled;
        self
    }

    /// Set the instance name
    pub fn instance_name(mut self, name: impl Into<String>) -> Self {
        self.config.instance_name = name.into();
        self
    }

    /// Set the data recorder to monitor
    pub fn recorder(mut self, recorder: Arc<DataRecorder>) -> Self {
        self.recorder = Some(recorder);
        self
    }

    /// Set the live feed bridge to monitor
    pub fn bridge(mut self, bridge: Arc<LiveFeedBridge>) -> Self {
        self.bridge = Some(bridge);
        self
    }

    /// Set the replay engine to monitor
    pub fn replay(mut self, replay: Arc<ReplayEngine>) -> Self {
        self.replay = Some(replay);
        self
    }

    /// Build the metrics server
    pub fn build(self) -> SimMetricsServer {
        let server = SimMetricsServer::new(self.config);

        if let Some(recorder) = self.recorder {
            server.set_recorder(recorder);
        }
        if let Some(bridge) = self.bridge {
            server.set_bridge(bridge);
        }
        if let Some(replay) = self.replay {
            server.set_replay(replay);
        }

        server
    }
}

impl Default for SimMetricsServerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Background task that periodically updates metrics from components
async fn metrics_collector_task(state: Arc<RwLock<ServerState>>, interval_ms: u64) {
    let interval = Duration::from_millis(interval_ms);
    let exporter = global_sim_metrics();

    loop {
        // Update metrics from components
        {
            let state = state.read();

            if let Some(ref recorder) = state.recorder {
                exporter.update_recorder_stats(&recorder.stats());
            }

            if let Some(ref bridge) = state.bridge {
                exporter.update_bridge_stats(&bridge.stats());
            }

            // Note: ReplayEngine metrics would be updated here when stats() is available
        }

        tokio::time::sleep(interval).await;
    }
}

/// Prometheus metrics handler
async fn metrics_handler() -> String {
    sim_prometheus_metrics()
}

/// Health check handler
async fn health_handler() -> &'static str {
    "OK"
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Start a simple metrics server with default configuration
///
/// This is a convenience function for quickly starting a metrics server.
///
/// # Example
///
/// ```rust,ignore
/// let handle = start_sim_metrics_server(9090).await?;
/// ```
pub async fn start_sim_metrics_server(
    port: u16,
) -> Result<SimMetricsServerHandle, MetricsServerError> {
    SimMetricsServer::new(SimMetricsServerConfig::new(port))
        .start()
        .await
}

/// Start a metrics server with recorder and bridge
///
/// This is a convenience function that sets up everything in one call.
///
/// # Example
///
/// ```rust,ignore
/// let handle = start_sim_metrics_server_with_components(
///     9090,
///     Some(recorder),
///     Some(bridge),
///     1000, // update interval ms
/// ).await?;
/// ```
pub async fn start_sim_metrics_server_with_components(
    port: u16,
    recorder: Option<Arc<DataRecorder>>,
    bridge: Option<Arc<LiveFeedBridge>>,
    update_interval_ms: u64,
) -> Result<SimMetricsServerHandle, MetricsServerError> {
    let mut builder = SimMetricsServer::builder()
        .port(port)
        .update_interval_ms(update_interval_ms);

    if let Some(r) = recorder {
        builder = builder.recorder(r);
    }

    if let Some(b) = bridge {
        builder = builder.bridge(b);
    }

    builder.build().start().await
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = SimMetricsServerConfig::default();
        assert_eq!(config.port, 9090);
        assert_eq!(config.host, "0.0.0.0");
        assert_eq!(config.update_interval_ms, 1000);
        assert!(config.try_alternative_ports);
    }

    #[test]
    fn test_config_builder() {
        let config = SimMetricsServerConfig::new(8080)
            .with_host("127.0.0.1")
            .with_update_interval(Duration::from_millis(500))
            .with_alternative_ports(false)
            .with_instance_name("test");

        assert_eq!(config.port, 8080);
        assert_eq!(config.host, "127.0.0.1");
        assert_eq!(config.update_interval_ms, 500);
        assert!(!config.try_alternative_ports);
        assert_eq!(config.instance_name, "test");
    }

    #[test]
    fn test_server_builder() {
        let server = SimMetricsServer::builder()
            .port(9091)
            .host("localhost")
            .update_interval_ms(2000)
            .instance_name("test-instance")
            .build();

        assert_eq!(server.config.port, 9091);
        assert_eq!(server.config.host, "localhost");
        assert_eq!(server.config.update_interval_ms, 2000);
        assert_eq!(server.config.instance_name, "test-instance");
    }

    #[tokio::test]
    async fn test_server_start_stop() {
        let server = SimMetricsServer::builder()
            .port(0) // Use OS-assigned port
            .try_alternative_ports(false)
            .build();

        let handle = server.start().await.expect("Failed to start server");

        assert!(handle.is_running());
        let port = handle.port();
        assert!(port > 0);

        // Give it a moment to start
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Shutdown
        handle.shutdown().await;
    }

    #[tokio::test]
    async fn test_metrics_endpoint() {
        let server = SimMetricsServer::builder()
            .port(0) // Use OS-assigned port
            .build();

        let handle = server.start().await.expect("Failed to start server");
        let addr = handle.address();

        // Give it a moment to start
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Make a request to the metrics endpoint
        let client = reqwest::Client::new();
        let response = client
            .get(format!("http://{}/metrics", addr))
            .send()
            .await
            .expect("Failed to send request");

        assert!(response.status().is_success());

        let body = response.text().await.expect("Failed to read response");
        // Should contain some Prometheus-format output (even if empty, headers should be there)
        assert!(body.is_empty() || body.contains('#') || body.contains("sim_"));

        handle.shutdown().await;
    }

    #[tokio::test]
    async fn test_health_endpoint() {
        let server = SimMetricsServer::builder()
            .port(0) // Use OS-assigned port
            .build();

        let handle = server.start().await.expect("Failed to start server");
        let addr = handle.address();

        // Give it a moment to start
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Make a request to the health endpoint
        let client = reqwest::Client::new();
        let response = client
            .get(format!("http://{}/health", addr))
            .send()
            .await
            .expect("Failed to send request");

        assert!(response.status().is_success());

        let body = response.text().await.expect("Failed to read response");
        assert_eq!(body, "OK");

        handle.shutdown().await;
    }
}
