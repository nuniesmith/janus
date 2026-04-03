//! Application state for the JANUS Rust Gateway.
//!
//! This module defines the shared application state that is accessible
//! across all route handlers via Axum's State extractor.

use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;

use crate::config::Settings;
use crate::metrics::GatewayMetrics;
use crate::rate_limit::EndpointRateLimiter;
use crate::redis_dispatcher::SignalDispatcher;

/// Placeholder for gRPC client to Janus services.
///
/// This will be replaced with a proper tonic client once the
/// proto definitions are compiled.
pub struct JanusGrpcClient {
    /// Forward service URL
    pub forward_url: String,
    /// Backward service URL
    pub backward_url: String,
    /// Connection state
    connected: bool,
}

impl JanusGrpcClient {
    /// Create a new gRPC client (not connected).
    pub fn new(forward_url: impl Into<String>, backward_url: impl Into<String>) -> Self {
        Self {
            forward_url: forward_url.into(),
            backward_url: backward_url.into(),
            connected: false,
        }
    }

    /// Connect to the Janus services.
    ///
    /// Connect to the Janus services.
    ///
    /// Currently a placeholder that marks the client as connected.
    /// To enable real gRPC communication, replace the body with tonic
    /// channel creation:
    /// ```ignore
    /// let channel = tonic::transport::Channel::from_shared(self.forward_url.clone())?
    ///     .connect().await?;
    /// ```
    pub async fn connect(&mut self) -> anyhow::Result<()> {
        info!(
            "Connecting to Janus services: forward={}, backward={}",
            self.forward_url, self.backward_url
        );

        // Placeholder — the forward service is reached via Redis signal
        // dispatch today; tonic channels will be added when direct gRPC
        // communication is needed (e.g., for bidirectional streaming).
        self.connected = true;

        Ok(())
    }

    /// Check if connected.
    #[allow(dead_code)]
    pub fn is_connected(&self) -> bool {
        self.connected
    }

    /// Close the connection.
    pub async fn close(&mut self) {
        self.connected = false;
        info!("Janus gRPC client disconnected");
    }
}

/// Application state shared across all handlers.
///
/// This struct holds all the shared resources needed by route handlers,
/// including database connections, Redis clients, and gRPC clients.
#[allow(dead_code)]
pub struct AppState {
    /// Application settings
    pub settings: Settings,

    /// Redis signal dispatcher for publishing to forward service
    pub signal_dispatcher: Arc<SignalDispatcher>,

    /// gRPC client for Janus services (Forward, Backward)
    ///
    /// Wrapped in RwLock to allow connection management
    pub janus_client: RwLock<Option<JanusGrpcClient>>,

    /// Heartbeat task handle
    ///
    /// Stored to allow graceful shutdown
    pub heartbeat_handle: RwLock<Option<tokio::task::JoinHandle<()>>>,

    /// Prometheus metrics
    pub metrics: Arc<GatewayMetrics>,

    /// Endpoint-specific rate limiters
    pub endpoint_rate_limiter: Arc<EndpointRateLimiter>,
}

impl AppState {
    /// Create a new application state.
    pub fn new(settings: Settings, signal_dispatcher: Arc<SignalDispatcher>) -> Self {
        let metrics = Arc::new(GatewayMetrics::new().expect("Failed to create metrics"));
        let endpoint_rate_limiter = Arc::new(EndpointRateLimiter::new());

        Self {
            settings,
            signal_dispatcher,
            janus_client: RwLock::new(None),
            heartbeat_handle: RwLock::new(None),
            metrics,
            endpoint_rate_limiter,
        }
    }

    /// Create application state with custom metrics and rate limiter.
    #[allow(dead_code)]
    pub fn with_metrics_and_limiter(
        settings: Settings,
        signal_dispatcher: Arc<SignalDispatcher>,
        metrics: Arc<GatewayMetrics>,
        endpoint_rate_limiter: Arc<EndpointRateLimiter>,
    ) -> Self {
        Self {
            settings,
            signal_dispatcher,
            janus_client: RwLock::new(None),
            heartbeat_handle: RwLock::new(None),
            metrics,
            endpoint_rate_limiter,
        }
    }

    /// Initialize the gRPC client and connect to Janus services.
    pub async fn connect_janus_client(&self) -> anyhow::Result<()> {
        let mut client = JanusGrpcClient::new(
            &self.settings.janus_forward_url,
            &self.settings.janus_backward_url,
        );

        client.connect().await?;

        let mut guard = self.janus_client.write().await;
        *guard = Some(client);

        Ok(())
    }

    /// Start the heartbeat task for Dead Man's Switch.
    pub async fn start_heartbeat(&self, dispatcher: Arc<SignalDispatcher>) {
        let interval = self.settings.heartbeat_interval_secs;
        let metrics = self.metrics.clone();

        let handle = tokio::spawn(async move {
            let mut interval_timer =
                tokio::time::interval(tokio::time::Duration::from_secs(interval));

            loop {
                interval_timer.tick().await;

                match dispatcher.send_heartbeat().await {
                    Ok(_) => {
                        metrics.record_heartbeat();
                    }
                    Err(e) => {
                        metrics.record_heartbeat_error();
                        tracing::warn!("Failed to send heartbeat: {}", e);
                    }
                }
            }
        });

        let mut guard = self.heartbeat_handle.write().await;
        *guard = Some(handle);

        info!("Heartbeat task started (interval: {}s)", interval);
    }

    /// Stop the heartbeat task.
    pub async fn stop_heartbeat(&self) {
        let mut guard = self.heartbeat_handle.write().await;
        if let Some(handle) = guard.take() {
            handle.abort();
            info!("Heartbeat task stopped");
        }
    }

    /// Graceful shutdown - close all connections.
    pub async fn shutdown(&self) {
        info!("Shutting down application state...");

        // Stop heartbeat
        self.stop_heartbeat().await;

        // Close signal dispatcher
        self.signal_dispatcher.close().await;

        // Close gRPC client
        let mut guard = self.janus_client.write().await;
        if let Some(mut client) = guard.take() {
            client.close().await;
        }

        info!("Application state shutdown complete");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_janus_client_creation() {
        let client = JanusGrpcClient::new("localhost:50051", "localhost:50052");
        assert_eq!(client.forward_url, "localhost:50051");
        assert_eq!(client.backward_url, "localhost:50052");
        assert!(!client.is_connected());
    }

    #[tokio::test]
    async fn test_app_state_creation() {
        let settings = Settings::from_env();
        let dispatcher = Arc::new(SignalDispatcher::new("redis://localhost:6379/0"));
        let state = AppState::new(settings, dispatcher);

        // Initially no client
        assert!(state.janus_client.read().await.is_none());
        assert!(state.heartbeat_handle.read().await.is_none());

        // Metrics should be initialized
        let encoded = state.metrics.encode();
        assert!(encoded.is_ok());
    }

    #[tokio::test]
    async fn test_app_state_with_custom_components() {
        let settings = Settings::from_env();
        let dispatcher = Arc::new(SignalDispatcher::new("redis://localhost:6379/0"));
        let metrics = Arc::new(GatewayMetrics::new().unwrap());
        let rate_limiter = Arc::new(EndpointRateLimiter::new());

        let state =
            AppState::with_metrics_and_limiter(settings, dispatcher, metrics.clone(), rate_limiter);

        // Should use the provided metrics
        assert!(Arc::ptr_eq(&state.metrics, &metrics));
    }
}
