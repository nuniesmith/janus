//! # janus-neuromorphic
//!
//! Brain-inspired neuromorphic computing modules for the Janus trading system.
//!
//! > ⚠️ **Experimental** — This crate is under active research and development.
//! > APIs are not yet stable. Brain-region modules (amygdala, basal_ganglia,
//! > cerebellum, hippocampus, etc.) represent computational analogues of
//! > biological structures and are subject to architectural changes.
//! >
//! > **Stabilization criteria:**
//! > - Live trading validation for at least 30 days
//! > - Per-module benchmark baselines established
//! > - Public API documented and approved
//! >
//! > See [`integration`] for the primary entry point once stabilized.

// ── Clippy suppressions ────────────────────────────────────────────────
// Only genuinely necessary suppressions are kept here. Mechanical lints
// (get_first, let_and_return, manual_clamp, vec_init_then_push, etc.)
// have been fixed in source.

// Structural — refactoring would require significant redesign
#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]
// Changing `&Vec<T>` / `&String` to `&[T]` / `&str` in public APIs is breaking
#![allow(clippy::ptr_arg)]
// SAFETY: The barrier() function in distributed/runtime.rs intentionally holds
// a Tokio Mutex guard across an .await point. The lock is used for coordination
// counting and is held only briefly; the alternative (drop + re-acquire) would
// introduce a TOCTOU race in the barrier protocol.
#![allow(clippy::await_holding_lock)]
// `is_multiple_of()` is nightly-only; `x % n == 0` is the stable idiom (~30 instances)
#![allow(clippy::manual_is_multiple_of)]
// `.map_or(false, ...)` / `.map_or(true, ...)` used throughout (~11 instances);
// replacing with `is_some_and` / `is_none_or` would raise MSRV
#![allow(clippy::unnecessary_map_or)]
// Sequential threshold checks (`if a { … } if b { … }`) are kept uncollapsed
// for readability in risk/safety code (~15 instances)
#![allow(clippy::collapsible_if)]
// Indexed `for i in 0..n` loops that access parallel arrays/slices by index
#![allow(clippy::needless_range_loop)]
// Single-arm matches kept intentionally for future extensibility
#![allow(clippy::single_match)]
// Several Default impls have all-default field values but use explicit
// construction for clarity (e.g. FearResponse, AnomalyResult)
#![allow(clippy::derivable_impls)]
// Test code and builders use `let mut x = T::default(); x.field = val;` extensively
#![allow(clippy::field_reassign_with_default)]
// `.is_some()` / `.is_ok()` checks followed by usage in stats tracking are
// intentional for clarity over `if let`
#![allow(clippy::unnecessary_unwrap)]
// Intentional bounds checks on values that may be unsigned zero
#![allow(clippy::absurd_extreme_comparisons)]
//!
//! ## Brain Regions
//!
//! - **Cortex**: Strategic planning and long-term memory (Manager Agent)
//! - **Hippocampus**: Episodic memory and experience replay (Worker Agent)
//! - **Basal Ganglia**: Action selection and reinforcement learning (Actor-Critic)
//! - **Thalamus**: Attention and multimodal fusion (Sensory Relay)
//! - **Prefrontal**: Logic, planning, and compliance (LTN)
//! - **Amygdala**: Fear, threat detection, and circuit breakers
//! - **Hypothalamus**: Homeostasis and risk appetite
//! - **Cerebellum**: Motor control and execution
//! - **Visual Cortex**: Pattern recognition and vision (GAF/ViViT)
//! - **Integration**: Brainstem and global coordination
//!
//! ## Service Connections
//!
//! The neuromorphic architecture connects to JANUS services via service bridges:
//!
//! - **Forward Service**: Receives signals for real-time execution
//! - **Backward Service**: Sends training data and receives historical analysis
//! - **CNS Service**: Reports health metrics and receives system commands
//! - **Data Service**: Receives market data for processing
//!
//! ## External Data Sources
//!
//! The thalamus module provides integration with external data sources:
//!
//! - **News**: Financial news and sentiment analysis
//! - **Weather**: Weather data for energy/agriculture markets
//! - **Celestial**: Moon phases, space weather, orbital data
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                    NEUROMORPHIC ARCHITECTURE                         │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │                                                                      │
//! │  ┌─────────────────────────────────────────────────────────────┐   │
//! │  │                     BRAIN REGIONS                            │   │
//! │  │                                                               │   │
//! │  │  ┌────────┐ ┌────────────┐ ┌───────────┐ ┌──────────────┐  │   │
//! │  │  │ Cortex │ │Hippocampus │ │  Thalamus │ │Basal Ganglia │  │   │
//! │  │  │(Plan)  │ │ (Memory)   │ │ (Sensory) │ │  (Action)    │  │   │
//! │  │  └────────┘ └────────────┘ └───────────┘ └──────────────┘  │   │
//! │  │                                                               │   │
//! │  │  ┌────────┐ ┌────────────┐ ┌───────────┐ ┌──────────────┐  │   │
//! │  │  │Amygdala│ │Hypothalamus│ │Cerebellum │ │ Visual Cortex│  │   │
//! │  │  │ (Fear) │ │  (Risk)    │ │ (Execute) │ │  (Pattern)   │  │   │
//! │  │  └────────┘ └────────────┘ └───────────┘ └──────────────┘  │   │
//! │  │                                                               │   │
//! │  │  ┌──────────────┐ ┌─────────────────────────────────────┐  │   │
//! │  │  │  Prefrontal  │ │           Integration               │  │   │
//! │  │  │  (Logic)     │ │  (Coordination + Service Bridges)   │  │   │
//! │  │  └──────────────┘ └─────────────────────────────────────┘  │   │
//! │  │                                                               │   │
//! │  └───────────────────────────┬───────────────────────────────────┘   │
//! │                              │                                       │
//! │                    ┌─────────▼─────────┐                            │
//! │                    │  Service Bridges  │                            │
//! │                    └─────────┬─────────┘                            │
//! └──────────────────────────────┼──────────────────────────────────────┘
//!                                │
//!          ┌─────────────────────┼─────────────────────┐
//!          │                     │                     │
//!     ┌────▼────┐          ┌─────▼────┐          ┌────▼────┐
//!     │ Forward │          │ Backward │          │   CNS   │
//!     │ Service │          │ Service  │          │ Service │
//!     └─────────┘          └──────────┘          └─────────┘
//! ```

// Re-export common types from the common crate
pub mod common {
    pub use common::{JanusError as Error, Result};
}

// Brain regions
pub mod amygdala;
pub mod basal_ganglia;
pub mod cerebellum;
pub mod cortex;
pub mod distributed;
pub mod gpu;
pub mod hippocampus;
pub mod hypothalamus;
pub mod integration;
pub mod prefrontal;
pub mod thalamus;
pub mod visual_cortex;

// Re-export key integration components for easy access
pub use integration::{
    // Service bridges
    BackwardBridgeConfig,
    BackwardServiceBridge,
    // Core coordination
    BrainCoordinator,
    BridgeManager,
    BridgeMessage,
    // Data pipeline
    Candle,
    CnsBridgeConfig,
    CnsServiceBridge,
    ConfigPayload,
    DataBridgeConfig,
    DataServiceBridge,
    ForwardBridgeConfig,
    ForwardServiceBridge,
    GafFeature,
    GlobalState,
    HealthPayload,
    // Training pipeline
    MarketDataLoader,
    MarketDataPayload,
    MarketDataPipeline,
    // Realtime system
    MarketDataPoint,
    MarketDataSample,
    MessageBus,
    NeuromorphicSignal,
    PipelineConfig,
    Priority,
    RealtimeConfig,
    RealtimeSystem,
    RiskLevel,
    ServiceBridge,
    SignalType,
    SystemCommand,
    SystemOrchestrator,
    TradingDecision,
    TradingSignal,
    TrainingBatch,
    TrainingConfig,
    TrainingMetrics,
    TrainingPayload,
    TrainingPipeline,
    TrainingSample,
};

// Re-export heterogeneity components for crowding resistance
pub use integration::{
    CrossInstanceMonitor, CrowdingAlert, HeterogeneityConfig, HeterogeneityProfile, MonitorConfig,
};

// Re-export key thalamus components for external data
pub use thalamus::{
    // Routing components
    Broadcast,
    // External data sources
    CelestialData,
    CelestialSource,
    ExternalDataAggregator,
    ExternalDataPoint,
    MoonPhase,
    NewsArticle,
    NewsSentiment,
    NewsSource,
    // Fusion components
    OrderbookFusion,
    Pathways,
    PriceFusion,
    Router,
    SentimentFusion,
    SpaceWeather,
    VolumeFusion,
    WeatherData,
    WeatherSource,
};
