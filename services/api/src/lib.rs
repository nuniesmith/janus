//! JANUS Rust Gateway Library
//!
//! High-performance API gateway for Project JANUS.
//!
//! This crate provides the core components for the Rust gateway service:
//!
//! - **config**: Environment-based configuration management
//! - **redis_dispatcher**: Redis Pub/Sub signal dispatch and heartbeat
//! - **routes**: Axum route handlers for REST API
//! - **state**: Shared application state
//! - **metrics**: Prometheus metrics collection and `/metrics` endpoint
//! - **rate_limit**: Request rate limiting using governor
//! - **grpc**: gRPC-Web support for browser clients via tonic-web
//!
//! # Architecture
//!
//! The gateway serves as the entry point for external clients (Web, Mobile)
//! and orchestrates communication with internal Rust services:
//!
//! ```text
//! ┌─────────────────┐     ┌─────────────────┐
//! │  Web Client     │     │  Mobile Client  │
//! │  (Kotlin/JS)    │     │  (KMP)          │
//! └────────┬────────┘     └────────┬────────┘
//!          │                       │
//!          │  HTTP/gRPC-Web        │
//!          ▼                       ▼
//! ┌─────────────────────────────────────────┐
//! │         janus-gateway (Rust)            │
//! │  • REST API (Axum)                      │
//! │  • gRPC-Web proxy (tonic-web)           │
//! │  • Redis Pub/Sub (signal dispatch)      │
//! │  • Dead Man's Switch (heartbeat)        │
//! │  • Prometheus metrics                   │
//! │  • Rate limiting                        │
//! └────────┬───────────────────┬────────────┘
//!          │                   │
//!          │ gRPC              │ Redis Pub/Sub
//!          ▼                   ▼
//! ┌─────────────────┐   ┌─────────────────┐
//! │  janus-forward  │   │  janus-backward │
//! │  (Live Trading) │   │  (Training)     │
//! └─────────────────┘   └─────────────────┘
//! ```
//!
//! # Example
//!
//! ```rust,no_run
//! use janus_gateway::{config::Settings, redis_dispatcher::SignalDispatcher};
//! use std::sync::Arc;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let settings = Settings::from_env();
//!     let dispatcher = Arc::new(SignalDispatcher::new(&settings.redis_signal_url));
//!     dispatcher.connect().await?;
//!
//!     // Dispatch a signal
//!     use janus_gateway::redis_dispatcher::Signal;
//!     let signal = Signal::new("BTCUSD", "Buy")
//!         .with_strength(0.8)
//!         .with_confidence(0.9);
//!     dispatcher.dispatch_signal(&signal).await?;
//!
//!     Ok(())
//! }
//! ```
//!
//! # Metrics
//!
//! The gateway exposes Prometheus metrics at `/metrics`:
//!
//! - `janus_gateway_http_requests_total` - Total HTTP requests by method, path, status
//! - `janus_gateway_http_request_duration_seconds` - Request duration histogram
//! - `janus_gateway_http_requests_in_flight` - Currently active requests
//! - `janus_gateway_signals_dispatched_total` - Signals sent by symbol and side
//! - `janus_gateway_redis_connected` - Redis connection status
//! - `janus_gateway_grpc_connected` - gRPC connection status
//! - `janus_gateway_heartbeats_sent_total` - Dead Man's Switch heartbeats
//! - `janus_gateway_uptime_seconds` - Service uptime
//!
//! # Rate Limiting
//!
//! Global and endpoint-specific rate limiting is provided via the `governor` crate:
//!
//! ```rust,no_run
//! use janus_gateway::rate_limit::{RateLimitConfig, RateLimitState};
//!
//! // Create a rate limiter: 100 req/s with burst of 50
//! let config = RateLimitConfig::new(100, 50);
//! let state = RateLimitState::new(config);
//!
//! // Check if request is allowed
//! match state.check() {
//!     Ok(_) => println!("Request allowed"),
//!     Err(wait_time) => println!("Rate limited, retry after {:?}", wait_time),
//! }
//! ```

pub mod config;
pub mod grpc;
pub mod metrics;
pub mod rate_limit;
pub mod redis_dispatcher;
pub mod routes;
pub mod state;

// Re-export commonly used types
pub use config::Settings;
pub use metrics::GatewayMetrics;
pub use rate_limit::{EndpointRateLimiter, RateLimitConfig, RateLimitState};
pub use redis_dispatcher::{Signal, SignalDispatcher};
pub use state::AppState;
