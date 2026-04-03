//! Hippocampus: Episodic Memory & Experience Replay

pub mod buffer;
pub mod consolidation;
pub mod diffusion;
pub mod emotional;
pub mod episodes;
pub mod memory;
pub mod replay;
pub mod swr;
pub mod worker;

pub use buffer::{EpisodicBuffer, Experience};
pub use consolidation::{ConsolidationStats, Consolidator};
pub use emotional::{
    EmotionalArousal, EmotionalMemory, EmotionalMemoryStats, EmotionalTag, EmotionalValence,
};
pub use memory::{MemoryEntry, MemoryType, VectorDB};
pub use replay::ReplayBuffer;
pub use worker::{Procedural, SkillLibrary, TacticalPolicy, WorkerAgent};

// Re-exports from diffusion module — Synthetic time series generation
pub use diffusion::{
    // Core DDPM types
    DdpmState,
    DiffusionConfig,
    DiffusionError,
    DiffusionTrainingStats,
    // Data types
    FeatureStatistics,
    GeneratedBatch,
    LossWeighting,
    // Noise schedule
    NoiseSchedule,
    NoiseScheduleType,
    NormalizationStats,
    // High-level generator
    QualityReport,
    RegimeLabel,
    SyntheticDataGenerator,
    TimeSeriesDDPM,
    TimeSeriesSample,
    VarianceType,
};

// Not yet implemented as a unified facade type:
// pub use swr::SWRSimulator;  — Sharp-Wave Ripple simulator facade
//
// Use the individual types directly instead:
//   swr::{BatchSampling, CompressedReplay, ConsolidationSync, RippleDetection}
