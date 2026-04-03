//! Integration: Brainstem & Global Coordination
//!
//! This module serves as the central coordination hub for the neuromorphic
//! architecture, connecting all brain regions and bridging to JANUS services.
//!
//! ## Components
//!
//! - **Coordinator**: Brain region coordination and message routing
//! - **Data**: Market data pipeline and preprocessing
//! - **Message Bus**: Inter-region communication
//! - **Orchestrator**: System-wide workflow orchestration
//! - **Realtime**: Real-time trading system integration
//! - **State**: Global state management
//! - **Training**: Training pipeline coordination
//! - **Service Bridges**: Connections to Forward/Backward/CNS/Data services
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    INTEGRATION LAYER                            │
//! │                  (Brainstem / Global Coordination)              │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                 │
//! │  ┌─────────────────────────────────────────────────────────┐    │
//! │  │                    Brain Regions                        │    │
//! │  │  ┌────────┐ ┌──────────┐ ┌─────────┐ ┌──────────────┐   │    │
//! │  │  │ Cortex │ │Hippocampus││Thalamus │ │Basal Ganglia │   │    │
//! │  │  └───┬────┘ └─────┬────┘ └────┬────┘ └──────┬───────┘   │    │
//! │  │      │            │           │             │           │    │
//! │  │      └────────────┴─────┬─────┴─────────────┘           │    │
//! │  │                         │                               │    │
//! │  │              ┌──────────▼──────────┐                    │    │
//! │  │              │    Message Bus      │                    │    │
//! │  │              │    (Coordinator)    │                    │    │
//! │  │              └──────────┬──────────┘                    │    │
//! │  └─────────────────────────┼────────────────────────────────┘   │
//! │                            │                                    │
//! │              ┌─────────────┼─────────────┐                      │
//! │              │             │             │                      │
//! │       ┌──────▼─────┐ ┌─────▼─────┐ ┌────▼─────┐                │
//! │       │   State    │ │Orchestrator│ │ Training │                │
//! │       │  Manager   │ │           │ │ Pipeline │                │
//! │       └──────┬─────┘ └─────┬─────┘ └────┬─────┘                │
//! │              │             │             │                      │
//! │              └─────────────┼─────────────┘                      │
//! │                            │                                    │
//! │              ┌─────────────▼─────────────┐                      │
//! │              │     Service Bridges       │                      │
//! │              └─────────────┬─────────────┘                      │
//! └────────────────────────────┼─────────────────────────────────────┘
//!                              │
//!          ┌───────────────────┼───────────────────┐
//!          │                   │                   │
//!     ┌────▼────┐        ┌────▼────┐        ┌────▼────┐
//!     │ Forward │        │ Backward │        │   CNS   │
//!     │ Service │        │ Service  │        │ Service │
//!     └─────────┘        └──────────┘        └─────────┘
//! ```

pub mod cognitive_core;
pub mod coordinator;
pub mod data;
pub mod heterogeneity;
pub mod message_bus;
pub mod orchestrator;
pub mod realtime;
pub mod service_bridges;
pub mod state;
pub mod training;

// Core coordination exports
pub use coordinator::{
    BrainCoordinator, BrainCoordinatorConfig, CoordinatorPhase, CoordinatorStats, RegionDescriptor,
};
pub use message_bus::{Message, MessageBus, MessageBusConfig, MessageBusStats};
pub use orchestrator::{
    OrchestratorConfig, OrchestratorPhase, OrchestratorStats, StepOutcome, StepRecord,
    SystemOrchestrator, WorkflowDefinition, WorkflowExecution, WorkflowOutcome, WorkflowStep,
};
pub use state::GlobalState;

// Data pipeline exports
pub use data::{
    Candle, GafFeature, MarketDataPipeline, PipelineConfig, TrainingBatch, TrainingSample,
};

// Realtime system exports
pub use realtime::{
    MarketDataPoint, RealtimeConfig, RealtimeSystem, TradingDecision, TradingSignal,
};

// Training pipeline exports
pub use training::{
    MarketDataLoader, MarketDataSample, TrainingConfig, TrainingMetrics, TrainingPipeline,
};

// Cognitive core exports
pub use cognitive_core::{
    CognitiveCore, CognitiveCoreConfig, ExperienceProcessingResult, FearResponse, SystemStats,
};

// Heterogeneity exports (engineered crowding resistance)
pub use heterogeneity::{
    BasalGangliaHeterogeneity, CerebellarHeterogeneity, CrossInstanceMonitor, CrowdingAlert,
    HeterogeneityConfig, HeterogeneityProfile, HippocampalHeterogeneity, HypothalamicHeterogeneity,
    MonitorConfig, PrefrontalHeterogeneity, ThalamicHeterogeneity,
};

// Service bridge exports
pub use service_bridges::{
    BackwardBridgeConfig, BackwardServiceBridge, BridgeManager, BridgeMessage, CnsBridgeConfig,
    CnsServiceBridge, ConfigPayload, DataBridgeConfig, DataServiceBridge, ForwardBridgeConfig,
    ForwardServiceBridge, HealthPayload, MarketDataPayload, NeuromorphicSignal, Priority,
    RiskLevel, ServiceBridge, SignalType, SystemCommand, TrainingPayload,
};
