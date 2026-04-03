#![allow(clippy::manual_range_contains)]
#![allow(clippy::redundant_closure)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::should_implement_trait)]

//! # Janus Logic
//!
//! The Neuro-Symbolic Reasoning Layer of Project JANUS.
//!
//! This crate provides two categories of logic implementations:
//!
//! ## 1. Non-Differentiable Logic (Traditional)
//!
//! Uses ndarray for inference-time constraint checking:
//! - `tnorm`: Traditional fuzzy logic operations (Łukasiewicz T-Norms)
//! - `constraints`: Rule-based constraint checking
//! - `signal`: Signal validation and filtering
//! - `position`: Position-related logic
//! - `risk`: Risk assessment logic
//! - `risk_engine`: Full risk evaluation engine
//!
//! ## 2. Differentiable Logic Tensor Networks (LTN)
//!
//! Uses Candle ML framework for gradient-based training:
//! - `diff_tnorm`: Differentiable fuzzy T-norms (Łukasiewicz, Product, Gödel)
//! - `predicates`: Learnable and threshold-based logical predicates
//! - `ltn`: Full Logic Tensor Network framework with rule composition and loss functions
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    Neuro-Symbolic System                    │
//! ├─────────────────────────────────────────────────────────────┤
//! │  Neural Network (ViViT)  →  Embeddings  →  Predictions      │
//! │           ↓                                    ↓             │
//! │      Grounding          ←─────────────────────┘             │
//! │           ↓                                                  │
//! │      Predicates         (learnable/threshold)               │
//! │           ↓                                                  │
//! │      Diff T-Norms       (AND, OR, IMPLIES, NOT)             │
//! │           ↓                                                  │
//! │      Logical Rules      (formulas, quantifiers)             │
//! │           ↓                                                  │
//! │   Satisfaction Loss     →  Backpropagation  →  Training     │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example: Training with Logical Constraints
//!
//! ```ignore
//! use logic::{DiffLTN, RuleBuilder, Grounding, TNormType};
//! use candle_core::{Device, Tensor};
//!
//! // Create LTN with Łukasiewicz t-norm
//! let mut ltn = DiffLTN::new(TNormType::Lukasiewicz);
//!
//! // Add logical rules
//! ltn.add_rule(
//!     RuleBuilder::new("high_conf_action")
//!         .implies("confidence_high", "action_allowed")
//!         .weight(1.0)
//!         .build()
//! );
//!
//! ltn.add_rule(
//!     RuleBuilder::new("risk_bounds")
//!         .requires("position_valid")
//!         .weight(2.0)
//!         .build()
//! );
//!
//! // Create grounding from neural network outputs
//! let mut grounding = Grounding::new();
//! grounding.set("confidence_high", confidence_tensor);
//! grounding.set("action_allowed", action_tensor);
//! grounding.set("position_valid", position_tensor);
//!
//! // Compute differentiable satisfaction loss
//! let logic_loss = ltn.satisfaction_loss(&grounding)?;
//!
//! // Combine with task loss for training
//! let total_loss = task_loss + logic_weight * logic_loss;
//! // Backpropagate...
//! ```

// =============================================================================
// Non-Differentiable Modules (ndarray-based)
// =============================================================================

pub mod constraints;
pub mod position;
pub mod risk;
pub mod risk_engine;
pub mod signal;
pub mod tnorm;

// =============================================================================
// Differentiable LTN Modules (Candle-based)
// =============================================================================

pub mod diff_tnorm;
pub mod ltn;
pub mod predicates;

// =============================================================================
// Re-exports: Traditional Logic
// =============================================================================

pub use constraints::*;
pub use position::*;
pub use risk::*;
pub use risk_engine::*;
pub use signal::*;
pub use tnorm::{FuzzyLogic, LogicTensorNetwork, LogicalRule as TraditionalLogicalRule};

// =============================================================================
// Re-exports: Differentiable LTN
// =============================================================================

// Core LTN types
pub use ltn::{
    DiffLTN, DiffLTNConfig, Formula, Grounding, LogicalRule, RuleBuilder, RuleSatisfaction,
    RuleSets, SatisfactionAggregation, SatisfactionReport,
};

// T-norm types and operations
pub use diff_tnorm::{DiffTNorm, DiffTNormConfig, TNormType, godel, lukasiewicz, product};

// Predicate types
pub use predicates::{
    LearnablePredicate, LearnablePredicateConfig, Predicate, SimilarityPredicate,
    SimilarityPredicateConfig, SimilarityType, ThresholdPredicate, ThresholdPredicateConfig,
    ThresholdType, TradingPredicates,
};
