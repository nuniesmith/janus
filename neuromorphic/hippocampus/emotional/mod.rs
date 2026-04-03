//! Emotional Memory Integration
//!
//! Bridges hippocampus (episodic memory) and amygdala (fear/threat detection)
//! to create emotionally-tagged memories for enhanced learning and consolidation.

pub mod emotional_tagging;

pub use emotional_tagging::{
    EmotionalArousal, EmotionalMemory, EmotionalMemoryStats, EmotionalTag, EmotionalValence,
};
