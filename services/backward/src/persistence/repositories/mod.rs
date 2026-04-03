//! # Repository Layer
//!
//! Data access layer for JANUS service providing:
//! - Signal repository for storing and querying trading signals
//! - Portfolio repository for portfolio management
//! - Position repository for position tracking
//! - Performance repository for analytics
//! - Metrics repository for risk and performance metrics

pub mod metrics_repository;
pub mod performance_repository;
pub mod portfolio_repository;
pub mod position_repository;
pub mod signal_repository;

pub use metrics_repository::MetricsRepository;
pub use performance_repository::PerformanceRepository;
pub use portfolio_repository::PortfolioRepository;
pub use position_repository::PositionRepository;
pub use signal_repository::SignalRepository;
