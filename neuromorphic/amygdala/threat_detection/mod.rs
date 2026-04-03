//! Threat Detection Component
//!
//! Part of Amygdala region

pub mod anomaly_detector;
pub mod black_swan;
pub mod correlation_break;
pub mod regime_shift;

// Re-exports
pub use anomaly_detector::AnomalyDetector;
pub use black_swan::BlackSwan;
pub use correlation_break::CorrelationBreak;
pub use regime_shift::RegimeShift;
