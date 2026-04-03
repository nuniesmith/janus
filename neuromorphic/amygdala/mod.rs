//! Amygdala: Fear, Threat Detection & Circuit Breakers

pub mod circuit_breakers;
pub mod fear;
pub mod fear_network;
pub mod kill_switch;
pub mod threat_detection;
pub mod vpin;

pub use fear::FearResponse;
pub use fear_network::{FearNetwork, MarketConditions, ThreatSignature};
pub use kill_switch::{EmergencyAction, KillSwitch};
pub use vpin::{VPINCalculator, VPINMonitor};

// Not yet implemented as unified facade types:
// pub use circuit_breakers::CircuitBreaker;
// pub use threat_detection::ThreatDetector;
//
// Use the individual types directly instead:
//   circuit_breakers::{CancelAll, KillSwitch, PositionFreeze, SafeMode}
//   threat_detection::{AnomalyDetector, BlackSwan, CorrelationBreak, RegimeShift}
