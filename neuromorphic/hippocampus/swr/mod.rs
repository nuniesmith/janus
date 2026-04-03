//! Swr Component
//!
//! Part of Hippocampus region

pub mod batch_sampling;
pub mod compressed_replay;
pub mod consolidation_sync;
pub mod ripple_detection;

// Re-exports
pub use batch_sampling::BatchSampling;
pub use compressed_replay::CompressedReplay;
pub use consolidation_sync::ConsolidationSync;
pub use ripple_detection::RippleDetection;
