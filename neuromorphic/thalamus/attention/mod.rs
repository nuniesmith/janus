//! Attention Component
//!
//! Part of Thalamus region

pub mod cross_attention;
pub mod focus;
pub mod gate;
pub mod saliency;

// Re-exports
pub use cross_attention::CrossAttention;
pub use focus::Focus;
pub use gate::Gate;
pub use saliency::Saliency;
