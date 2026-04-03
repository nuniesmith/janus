//! Basal Ganglia: Action Selection & Reinforcement Learning
//!
//! The Basal Ganglia region handles action selection, reinforcement learning,
//! and decision making. Key components include:
//!
//! - **Actor-Critic**: Actor-critic architecture for policy learning
//! - **Critic**: Value estimation and advantage computation
//! - **Direct Pathway**: "Go" pathway for action initiation
//! - **Indirect Pathway**: "NoGo" pathway for action inhibition
//! - **Reward**: Reward signal computation and shaping

pub mod actor_critic;
pub mod critic;
pub mod direct_pathway;
pub mod indirect_pathway;
pub mod praxeological;
pub mod reward;
pub mod selection;

// Re-exports - Actor-Critic
pub use actor_critic::{
    ActorCritic, Experience, PolicyNetwork, ValueNetwork as ActorCriticValueNetwork,
};

// Re-exports - Critic components
pub use critic::{
    Advantage, AdvantageBuilder, AdvantageConfig, AdvantageStats, DenseLayer, Gae, GaeBuilder,
    GaeConfig, GaePresets, GaeStats, TdLearning, Trajectory, Transition, ValueNetwork,
    ValueNetworkBuilder, ValueNetworkConfig,
};

// Re-exports - Pathways
pub use direct_pathway::DirectPathway;
pub use indirect_pathway::IndirectPathway;

// Re-exports - Reward
pub use reward::RewardCalculator;

// Re-exports - Praxeological
pub use praxeological::{Confidence, GoSignal, NoGoSignal, Threshold};

// Re-exports - Selection
pub use selection::{HabitCache, WinnerTakeAll};
