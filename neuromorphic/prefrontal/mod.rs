//! Prefrontal Cortex: Logic, Planning & Compliance

pub mod conscience;
pub mod goals;
pub mod ltn;
pub mod planning;
pub mod predicates;

pub use predicates::{MarketState, TradingPredicate};

// Not yet implemented as unified facade types:
// pub use ltn::LTNEngine;            — Logic Tensor Network reasoning engine facade
// pub use conscience::Conscience;     — unified compliance conscience facade
// pub use goals::GoalManager;         — unified goal lifecycle manager facade
// pub use planning::ExecutionPlanner; — unified execution planning facade
//
// Use the individual types directly instead:
//   ltn::{ComplianceScore, ConstraintSolver, FuzzyLogic, Predicates}
//   conscience::{PropFirmRules, RiskLimits}
//   goals::{Achievement, Priority, Revision}
//   planning::{Contingency, GoalDecomposition, PlanSynthesis, SubgoalGeneration}
