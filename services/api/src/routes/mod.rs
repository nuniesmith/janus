//! Route modules for the JANUS Rust Gateway.
//!
//! This module re-exports all route handlers for the gateway service.

pub mod dashboard;
pub mod health;
pub mod setup;
pub mod signals;

pub use dashboard::dashboard_routes;
pub use health::health_routes;
pub use setup::setup_routes;
pub use signals::signal_routes;

// Re-export metrics routes from the metrics module
pub use crate::metrics::metrics_routes;
