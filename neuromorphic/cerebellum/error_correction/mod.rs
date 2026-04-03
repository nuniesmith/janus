//! Error Correction Component
//!
//! Part of Cerebellum region

pub mod adaptive_correction;
pub mod execution_error;
pub mod feedback_loop;
pub mod pid_controller;

// Re-exports
pub use adaptive_correction::AdaptiveCorrection;
pub use execution_error::ExecutionError;
pub use feedback_loop::FeedbackLoop;
pub use pid_controller::PidController;
