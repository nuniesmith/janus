//! Cortex Module - Strategic Planning & Long-term Memory
//!
//! Implements the Manager Agent (Feudal RL) and long-term knowledge storage.
//! Corresponds to the cortical regions responsible for executive function.

pub mod manager;
pub mod memory;
pub mod planning;

pub use manager::StrategicPolicy;
pub use memory::Consolidation;
pub use planning::ScenarioAnalysis;

// Not yet implemented:
// pub use memory::MemoryConsolidation;  — unified memory consolidation facade
//
// Use the individual types directly instead:
//   memory::{Consolidation, Declarative, KnowledgeBase, Schemas}
//   planning::{Contingency, MonteCarlo, Optimization, ScenarioAnalysis}
