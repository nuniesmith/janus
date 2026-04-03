//! Cerebellum: Motor Control & Execution
//!
//! The Cerebellum region handles execution optimization and market microstructure
//! modeling. Key components include:
//!
//! - **Motor Control**: Order execution and position management
//! - **Forward Models**: Predictive models for market dynamics
//! - **Error Correction**: PID control and adaptive adjustments
//! - **Learning**: Adaptive execution improvement
//! - **Impact Models**: Almgren-Chriss optimal execution
//! - **Fill Probability**: Predict limit order fill likelihood
//! - **Adverse Selection**: Detect toxic order flow and informed trading
//! - **LOB Simulator**: Full limit order book simulator with price-time priority matching

pub mod adverse_selection;
pub mod almgren_chriss;
pub mod control;
pub mod error_correction;
pub mod fill_probability;
pub mod forward_model;
pub mod forward_models;
pub mod impact;
pub mod learning;
pub mod lob_simulator;

// Re-exports - Core components
pub use almgren_chriss::AlmgrenChriss;
pub use control::MotorControl;
pub use forward_model::ForwardModel;
pub use learning::ErrorCorrection;

// Re-exports - Fill Probability
pub use fill_probability::{
    FillFeatures, FillProbabilityConfig, FillProbabilityModel, FillProbabilityModelBuilder,
    FillProbabilityStats, FillRecord, OrderBookSnapshot, RecentTrade,
};

// Re-exports - Adverse Selection
pub use adverse_selection::{
    AdverseSelectionConfig, AdverseSelectionMetrics, AdverseSelectionModel,
    AdverseSelectionModelBuilder, AdverseSelectionPresets, AdverseSelectionStats, FillInfo,
    ToxicityLevel, TradeEvent,
};

// Re-exports - LOB Simulator
pub use lob_simulator::{
    // Fills & trades
    CancelReceipt,
    Fill,
    // Snapshots
    L2Level,
    L2Snapshot,
    L3Level,
    L3Order,
    L3Snapshot,
    // Core engine
    LimitOrderBook,
    LobConfig,
    LobError,
    // Analytics
    LobStatistics,
    MarketImpactModel,
    ModifyReceipt,
    // Order types
    Order,
    // Events
    OrderEvent,
    OrderEventType,
    OrderReceipt,
    OrderStatus,
    OrderType,
    Side,
    TimeInForce,
};
