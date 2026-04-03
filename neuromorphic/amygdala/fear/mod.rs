//! Fear Component
//!
//! Part of Amygdala region

pub mod emotional_memory;
pub mod fear_network;
pub mod panic_detection;
pub mod threat_response;

// Re-exports
pub use emotional_memory::EmotionalMemory;
pub use fear_network::{FearNetwork, FearResponse};
pub use panic_detection::PanicDetection;
pub use threat_response::ThreatResponse;
