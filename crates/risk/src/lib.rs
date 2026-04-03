//! Risk Management Module
//!
//! This module provides comprehensive risk management tools for trading,
//! including position sizing, profit allocation, multi-TP/SL strategies,
//! account scaling recommendations, and cross-asset correlation tracking.

pub mod correlation;
pub mod position_sizer;

// Re-exports for convenient usage
pub use position_sizer::{
    AccountTier, AccountTierInfo, Direction, MultiTpSlStrategy, PositionSizeResult, PositionSizer,
    ProfitAllocation, ProfitAllocator, ScalingRecommendation, TpLevel,
};

pub use correlation::{CorrelationConfig, CorrelationError, CorrelationTracker};
