//! Critic Component
//!
//! Part of Basal Ganglia region
//!
//! The critic evaluates states and actions to guide policy learning.
//! Key components:
//!
//! - **ValueNetwork**: Neural network for state value estimation V(s)
//! - **Advantage**: Computes advantage A(s,a) = Q(s,a) - V(s)
//! - **TdLearning**: Temporal difference learning algorithms
//! - **GAE**: Generalized Advantage Estimation for variance reduction

pub mod advantage;
pub mod gae;
pub mod td_learning;
pub mod value_network;

// Re-exports - Value Network
pub use value_network::{DenseLayer, ValueNetwork, ValueNetworkBuilder, ValueNetworkConfig};

// Re-exports - Advantage
pub use advantage::{Advantage, AdvantageBuilder, AdvantageConfig, AdvantageStats, Transition};

// Re-exports - TD Learning
pub use td_learning::TdLearning;

// Re-exports - GAE
pub use gae::{Gae, GaeBuilder, GaeConfig, GaePresets, GaeStats, Trajectory};
