//! Episodes Component
//!
//! Part of Hippocampus region
//!
//! This module contains episodic memory components for the hippocampus:
//! - Trade episodes: Complete trade sequences from entry to exit
//! - Market events: Significant market events detection and storage
//! - Regime transitions: Market regime change tracking
//! - Spatial map: Market state space representation

pub mod market_events;
pub mod regime_transitions;
pub mod spatial_map;
pub mod trade_episodes;

// Re-exports - Main types
pub use market_events::MarketEvents;
pub use regime_transitions::RegimeTransitions;
pub use spatial_map::SpatialMap;
pub use trade_episodes::TradeEpisodes;

// Re-exports - Trade episode types
pub use trade_episodes::{
    EntrySignal, ExitReason, MarketContext, MarketRegime as TradeMarketRegime, TradeDirection,
    TradeEpisode, TradeOutcome, TradePattern, TradeStatistics,
};

// Re-exports - Market event types
pub use market_events::{
    DataPoint, DetectionThresholds, EventSeverity, EventType, MarketEvent, MarketEventsConfig,
    MarketEventsStats, SymbolStats,
};

// Re-exports - Regime transition types
pub use regime_transitions::{
    MarketRegime, RegimeTransition, RegimeTransitionsConfig, TransitionCause, TransitionMatrix,
    TransitionStats,
};

// Re-exports - Spatial map types
pub use spatial_map::{
    PlaceCell, SpatialMapConfig, SpatialMapStats, StateDimension, StatePoint, StateTrajectory,
    TrajectoryStats,
};
