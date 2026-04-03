//! API Module for FKS Execution Service
//!
//! This module provides both gRPC and HTTP interfaces for the execution service.
//!
//! # gRPC API
//!
//! The gRPC API implements the `ExecutionService` defined in `execution.proto`:
//! - `SubmitSignal` - Submit trading signals for execution
//! - `GetOrderStatus` - Query order status
//! - `GetActiveOrders` - List active orders
//! - `CancelOrder` - Cancel a specific order
//! - `CancelAllOrders` - Cancel all orders (with filters)
//! - `GetPositions` - Get current positions
//! - `GetAccount` - Get account information
//! - `StreamUpdates` - Stream real-time execution updates
//! - `HealthCheck` - Service health check
//!
//! # HTTP API
//!
//! The HTTP API provides REST endpoints for administration and monitoring:
//! - `GET /health` - Overall health check
//! - `GET /health/ready` - Readiness probe
//! - `GET /health/live` - Liveness probe
//! - `GET /metrics` - Prometheus metrics
//! - `GET /api/v1/orders` - List orders
//! - `GET /api/v1/orders/:id` - Get order details
//! - `POST /api/v1/orders/:id/cancel` - Cancel order
//! - `POST /api/v1/orders/cancel-all` - Cancel all orders
//! - `GET /api/v1/stats` - Service statistics
//! - `GET /api/v1/admin/config` - Configuration
//!
//! # Example
//!
//! ```ignore
//! use janus_execution::api::{grpc::ExecutionServiceImpl, http};
//! use janus_execution::orders::OrderManager;
//! use std::sync::Arc;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create order manager
//!     let order_manager = Arc::new(OrderManager::new("localhost:9009").await?);
//!
//!     // Create gRPC service
//!     let grpc_service = ExecutionServiceImpl::new(order_manager.clone());
//!     let grpc_server = grpc_service.into_server();
//!
//!     // Create HTTP router
//!     let http_state = http::HttpState {
//!         order_manager: order_manager.clone(),
//!     };
//!     let http_router = http::create_router(http_state);
//!
//!     // Start servers...
//!     Ok(())
//! }
//! ```

pub mod grpc;
pub mod http;

// Re-export commonly used types
pub use grpc::ExecutionServiceImpl;
pub use http::{HttpState, create_router as create_http_router};
