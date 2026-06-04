//! Compliance utilities.
//!
//! Currently this crate provides wash-sale detection ([`WashSaleDetector`]).
//! The prop-firm rule checking that used to live here (`ComplianceSheriff`)
//! was consolidated onto `janus_models::prop_firm::PropFirmValidator`.

pub mod wash_sale;

// Re-exports
pub use wash_sale::{
    Trade, TradeAction, WashSaleDetector, WashSaleError, WashSaleResult, WashSaleStats,
};
