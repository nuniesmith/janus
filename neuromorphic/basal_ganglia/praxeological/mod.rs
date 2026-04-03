//! Praxeological Component
//!
//! Part of Basal Ganglia region

pub mod confidence;
pub mod go_signal;
pub mod no_go_signal;
pub mod threshold;

// Re-exports
pub use confidence::Confidence;
pub use go_signal::GoSignal;
pub use no_go_signal::NoGoSignal;
pub use threshold::Threshold;
