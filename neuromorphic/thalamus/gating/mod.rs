//! Gating Component
//!
//! Part of Thalamus region
//!
//! Implements thalamic sensory gating mechanisms that control which market
//! signals are admitted for further processing. Combines static threshold-based
//! filtering with dynamic Wilson-Cowan oscillatory gating.
//!
//! ## Components
//!
//! - **Filter**: Noise filtering and signal-to-noise enhancement
//! - **Relevance**: Relevance scoring for incoming signals
//! - **SensoryGate**: Multi-dimensional signal admission control
//! - **Threshold**: Adaptive threshold management
//! - **WilsonCowan**: Oscillatory gating via coupled E/I population dynamics

pub mod filter;
pub mod relevance;
pub mod sensory_gate;
pub mod threshold;
pub mod wilson_cowan;

// Re-exports
pub use filter::Filter;
pub use relevance::Relevance;
pub use sensory_gate::SensoryGate;
pub use threshold::Threshold;
pub use wilson_cowan::{
    BifurcationPoint, BifurcationType, ExternalDrive, FixedPointStability, GatingDrive, Integrator,
    NeuromodulatorInput, OscillationMode, OscillatorState, WilsonCowan, WilsonCowanConfig,
    WilsonCowanStats,
};
