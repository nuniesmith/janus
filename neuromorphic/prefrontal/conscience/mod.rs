//! Conscience Component
//!
//! Part of Prefrontal region

pub mod position_limits;
pub mod prop_firm_rules;
pub mod risk_limits;
pub mod wash_sale;

// Re-exports
pub use position_limits::PositionLimits;
pub use prop_firm_rules::PropFirmRules;
pub use risk_limits::RiskLimits;
pub use wash_sale::WashSale;
