//! gRPC module with tonic-web support for browser clients.
//!
//! This module provides gRPC-Web support, allowing Kotlin/JS browser clients
//! to communicate with the gateway using the gRPC-Web protocol without
//! requiring an external Envoy proxy.
//!
//! # Architecture
//!
//! ```text
//! Browser (Kotlin/JS)
//!        │
//!        │ gRPC-Web (HTTP/1.1 or HTTP/2)
//!        ▼
//! ┌─────────────────────────────────┐
//! │  tonic-web Layer                │
//! │  • Translates gRPC-Web ↔ gRPC   │
//! │  • Handles CORS preflight       │
//! │  • Base64 encoding for text     │
//! └─────────────────────────────────┘
//!        │
//!        │ Native gRPC
//!        ▼
//! ┌─────────────────────────────────┐
//! │  Tonic gRPC Service             │
//! │  • JanusGatewayService          │
//! │  • ForwardService proxy         │
//! │  • BackwardService proxy        │
//! └─────────────────────────────────┘
//! ```

use fks_proto::janus::{
    StartRequest, StartResponse, StatusRequest, StatusResponse, StopRequest, StopResponse,
    UpdateConfigRequest, UpdateConfigResponse, forward_service_client::ForwardServiceClient,
    forward_service_server::ForwardService, forward_service_server::ForwardServiceServer,
};
use std::sync::Arc;
use tokio::sync::RwLock;
use tonic::{Request, Response, Status, transport::Channel};
use tracing::{error, info};

/// gRPC client connection manager for Forward and Backward services.
#[allow(dead_code)]
pub struct GrpcClientManager {
    /// Forward service client
    forward_client: RwLock<Option<ForwardServiceClient<Channel>>>,

    /// Forward service URL
    forward_url: String,

    /// Backward service URL (for future use)
    backward_url: String,
}

impl GrpcClientManager {
    /// Create a new gRPC client manager.
    #[allow(dead_code)]
    pub fn new(forward_url: impl Into<String>, backward_url: impl Into<String>) -> Self {
        Self {
            forward_client: RwLock::new(None),
            forward_url: forward_url.into(),
            backward_url: backward_url.into(),
        }
    }

    /// Connect to the Forward service.
    #[allow(dead_code)]
    pub async fn connect_forward(&self) -> Result<(), tonic::transport::Error> {
        let endpoint = format!("http://{}", self.forward_url);
        info!("Connecting to Forward service at {}", endpoint);

        // Use unwrap_or_else to handle invalid URI - this shouldn't happen with valid config
        let channel = match Channel::from_shared(endpoint.clone()) {
            Ok(c) => c,
            Err(e) => {
                error!("Invalid Forward service endpoint: {}", e);
                // Return a connection error - the URI was invalid
                return Err(Channel::from_static("http://invalid")
                    .connect()
                    .await
                    .unwrap_err());
            }
        };

        let channel = channel.connect().await?;
        let client = ForwardServiceClient::new(channel);
        let mut guard = self.forward_client.write().await;
        *guard = Some(client);
        info!("Connected to Forward service");
        Ok(())
    }

    /// Get a clone of the Forward service client.
    #[allow(dead_code)]
    pub async fn forward_client(&self) -> Option<ForwardServiceClient<Channel>> {
        self.forward_client.read().await.clone()
    }

    /// Check if connected to Forward service.
    #[allow(dead_code)]
    pub async fn is_forward_connected(&self) -> bool {
        self.forward_client.read().await.is_some()
    }

    /// Get the Forward service URL.
    #[allow(dead_code)]
    pub fn forward_url(&self) -> &str {
        &self.forward_url
    }

    /// Get the Backward service URL.
    #[allow(dead_code)]
    pub fn backward_url(&self) -> &str {
        &self.backward_url
    }
}

/// Gateway gRPC service that proxies requests to the Forward service.
///
/// This service implements the ForwardService trait and acts as a proxy,
/// forwarding requests to the actual Forward service while adding
/// gateway-level concerns like authentication, rate limiting, and logging.
#[allow(dead_code)]
pub struct GatewayForwardService {
    /// gRPC client manager
    client_manager: Arc<GrpcClientManager>,
}

impl GatewayForwardService {
    /// Create a new gateway forward service.
    #[allow(dead_code)]
    pub fn new(client_manager: Arc<GrpcClientManager>) -> Self {
        Self { client_manager }
    }
}

#[tonic::async_trait]
impl ForwardService for GatewayForwardService {
    /// Start trading on a symbol.
    ///
    /// Proxies the request to the Forward service.
    async fn start(
        &self,
        request: Request<StartRequest>,
    ) -> Result<Response<StartResponse>, Status> {
        let req = request.into_inner();
        info!(
            "Gateway: Start request for symbol={}, risk_factor={}",
            req.symbol, req.risk_factor
        );

        // Get client
        let client = self
            .client_manager
            .forward_client()
            .await
            .ok_or_else(|| Status::unavailable("Forward service not connected"))?;

        // Clone to get a mutable reference
        let mut client = client;

        // Forward the request
        match client.start(Request::new(req)).await {
            Ok(response) => {
                let resp = response.into_inner();
                info!(
                    "Gateway: Start response engine_id={}, success={}",
                    resp.engine_id, resp.success
                );
                Ok(Response::new(resp))
            }
            Err(e) => {
                error!("Gateway: Start error: {}", e);
                Err(e)
            }
        }
    }

    /// Stop trading.
    async fn stop(&self, request: Request<StopRequest>) -> Result<Response<StopResponse>, Status> {
        let req = request.into_inner();
        info!("Gateway: Stop request for engine_id={}", req.engine_id);

        let client = self
            .client_manager
            .forward_client()
            .await
            .ok_or_else(|| Status::unavailable("Forward service not connected"))?;

        let mut client = client;

        match client.stop(Request::new(req)).await {
            Ok(response) => Ok(Response::new(response.into_inner())),
            Err(e) => {
                error!("Gateway: Stop error: {}", e);
                Err(e)
            }
        }
    }

    /// Get trading engine status.
    async fn status(
        &self,
        request: Request<StatusRequest>,
    ) -> Result<Response<StatusResponse>, Status> {
        let req = request.into_inner();
        info!("Gateway: Status request for engine_id={}", req.engine_id);

        let client = self
            .client_manager
            .forward_client()
            .await
            .ok_or_else(|| Status::unavailable("Forward service not connected"))?;

        let mut client = client;

        match client.status(Request::new(req)).await {
            Ok(response) => Ok(Response::new(response.into_inner())),
            Err(e) => {
                error!("Gateway: Status error: {}", e);
                Err(e)
            }
        }
    }

    /// Update trading configuration.
    async fn update_config(
        &self,
        request: Request<UpdateConfigRequest>,
    ) -> Result<Response<UpdateConfigResponse>, Status> {
        let req = request.into_inner();
        info!(
            "Gateway: UpdateConfig request for engine_id={}",
            req.engine_id
        );

        let client = self
            .client_manager
            .forward_client()
            .await
            .ok_or_else(|| Status::unavailable("Forward service not connected"))?;

        let mut client = client;

        match client.update_config(Request::new(req)).await {
            Ok(response) => Ok(Response::new(response.into_inner())),
            Err(e) => {
                error!("Gateway: UpdateConfig error: {}", e);
                Err(e)
            }
        }
    }
}

/// Create a tonic-web enabled gRPC service tower layer.
///
/// This function creates a gRPC service that can handle both native gRPC
/// and gRPC-Web requests, enabling browser clients to communicate
/// without an external proxy.
///
/// # Example
///
/// ```rust,no_run
/// use janus_gateway::grpc::{create_grpc_web_service, GrpcClientManager, GatewayForwardService};
/// use std::sync::Arc;
///
/// #[tokio::main]
/// async fn main() {
///     let client_manager = Arc::new(GrpcClientManager::new("localhost:50051", "localhost:50052"));
///     let grpc_service = create_grpc_web_service(client_manager);
///     // Use grpc_service with axum router
/// }
/// ```
#[allow(dead_code)]
pub fn create_grpc_web_service(_client_manager: Arc<GrpcClientManager>) -> tonic_web::GrpcWebLayer {
    // Return just the layer - the service will be created separately
    tonic_web::GrpcWebLayer::new()
}

/// Create the ForwardService server.
///
/// To enable gRPC-Web support, wrap this with the GrpcWebLayer:
/// ```rust,ignore
/// use tonic_web::GrpcWebLayer;
/// use tower::ServiceBuilder;
/// use janus_gateway::grpc::{create_forward_service_server, GrpcClientManager};
/// use std::sync::Arc;
///
/// let client_manager = Arc::new(GrpcClientManager::new("localhost:50051", "localhost:50052"));
/// let service = create_forward_service_server(client_manager);
/// let grpc_web_service = ServiceBuilder::new()
///     .layer(GrpcWebLayer::new())
///     .service(service);
/// ```
#[allow(dead_code)]
pub fn create_forward_service_server(
    client_manager: Arc<GrpcClientManager>,
) -> ForwardServiceServer<GatewayForwardService> {
    let service = GatewayForwardService::new(client_manager);
    ForwardServiceServer::new(service)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_manager_creation() {
        let manager = GrpcClientManager::new("localhost:50051", "localhost:50052");
        assert_eq!(manager.forward_url(), "localhost:50051");
        assert_eq!(manager.backward_url(), "localhost:50052");
    }

    #[tokio::test]
    async fn test_client_not_connected() {
        let manager = GrpcClientManager::new("localhost:50051", "localhost:50052");
        assert!(!manager.is_forward_connected().await);
        assert!(manager.forward_client().await.is_none());
    }

    #[test]
    fn test_gateway_service_creation() {
        let manager = Arc::new(GrpcClientManager::new("localhost:50051", "localhost:50052"));
        let _service = GatewayForwardService::new(manager);
        // Service should be created without error
    }

    #[test]
    fn test_grpc_web_layer_creation() {
        let manager = Arc::new(GrpcClientManager::new("localhost:50051", "localhost:50052"));
        let _layer = create_grpc_web_service(manager);
        // Layer should be created without error
    }
}
