//! Risk Appetite Management
//!
//! Part of the Hypothalamus region.
//! Manages risk appetite through multiple sub-components:
//!
//! - **appetite**: Core risk appetite state and computation
//! - **manager**: High-level risk management coordination
//! - **adaptation**: Dynamic risk appetite adaptation based on performance
//! - **appetite_curve**: Risk appetite curve modeling (utility functions)
//! - **confidence**: Risk-adjusted confidence estimation
//! - **fear_greed_index**: Composite fear/greed market sentiment index

pub mod adaptation;
pub mod appetite;
pub mod appetite_curve;
pub mod confidence;
pub mod fear_greed_index;
pub mod manager;

#[cfg(test)]
mod tests;

// Re-exports
pub use appetite::RiskAppetite;
pub use appetite_curve::AppetiteCurve;
pub use confidence::Confidence as RiskConfidence;
pub use fear_greed_index::{FearGreedIndex, FearGreedRegime};
pub use manager::RiskManager;
