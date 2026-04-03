//! Manager Component
//!
//! Part of Cortex region

pub mod goal_setting;
pub mod hierarchical_rl;
pub mod strategic_policy;
pub mod subgoal_generation;

// Re-exports
pub use goal_setting::GoalSetting;
pub use hierarchical_rl::HierarchicalRl;
pub use strategic_policy::StrategicPolicy;
pub use subgoal_generation::SubgoalGeneration;
