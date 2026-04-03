//! # Janus Core
//!
//! The canonical data model for Project JANUS.
//! This crate defines the "Rosetta Stone" types that ensure bitwise
//! compatibility between Forward and Backward services.
//!
//! All domain types use zero-copy serialization (rkyv) for maximum
//! performance in the hot path.

pub mod errors;
pub mod risk;
pub mod traits;
pub mod types;

pub use errors::*;
pub use risk::{
    Order as RiskOrder, PortfolioState, Position, RiskChecker, RiskError, RiskLimits,
    RiskUtilization,
};
pub use traits::*;
pub use types::*;
