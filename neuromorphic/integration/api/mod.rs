//! Api Component
//!
//! Part of Integration region

pub mod rest_api;
pub mod grpc_server;
pub mod websocket;
pub mod graphql;

// Re-exports
pub use rest_api::RestApi;
pub use grpc_server::GrpcServer;
pub use websocket::Websocket;
pub use graphql::Graphql;
