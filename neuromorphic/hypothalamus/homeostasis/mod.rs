//! Homeostasis Component
//!
//! Part of Hypothalamus region

pub mod balance_tracker;
pub mod controller;
pub mod correction;
pub mod deviation_detector;
pub mod setpoint;

// Re-exports
pub use balance_tracker::BalanceTracker;
pub use controller::HomeostasisController;
pub use correction::Correction;
pub use deviation_detector::DeviationDetector;
pub use setpoint::Setpoint;
