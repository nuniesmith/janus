//! Logic Tensor Network (LTN) Module for Project JANUS
//!
//! This module implements a neuro-symbolic reasoning engine that combines:
//! - **Neural Learning**: Deep learning from market data
//! - **Symbolic Reasoning**: Domain knowledge encoded as logical axioms
//! - **Fuzzy Logic**: Differentiable operators for gradient-based learning
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────┐
//! │                    DSP Pipeline (8D Features)                │
//! └────────────────────────┬─────────────────────────────────────┘
//!                          │
//!                          ▼
//! ┌──────────────────────────────────────────────────────────────┐
//! │                   LTN Neural Core                            │
//! │  ┌────────────────────────────────────────────────────────┐  │
//! │  │  Neural Network: 8 → 32 → 64 → 32 → 3                │  │
//! │  │  Output: [P(long), P(neutral), P(short)]              │  │
//! │  └────────────────────┬───────────────────────────────────┘  │
//! │                       │                                      │
//! │                       ▼                                      │
//! │  ┌────────────────────────────────────────────────────────┐  │
//! │  │  Axiom Evaluation (10 market rules)                   │  │
//! │  │  • Trending + div → Long/Short                        │  │
//! │  │  • Mean-reverting + div → Fade                        │  │
//! │  │  • Low confidence → Neutral                           │  │
//! │  │  • High noise → Caution                               │  │
//! │  └────────────────────┬───────────────────────────────────┘  │
//! │                       │                                      │
//! │                       ▼                                      │
//! │  ┌────────────────────────────────────────────────────────┐  │
//! │  │  Hybrid Loss = α·L_supervised + (1-α)·L_semantic      │  │
//! │  └────────────────────────────────────────────────────────┘  │
//! └──────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use janus_ltn::{LtnConfig, axioms::AxiomLibrary, predicates::TradingSignal};
//!
//! // Create configuration
//! let config = LtnConfig::default();
//!
//! // Initialize axiom library
//! let axioms = AxiomLibrary::default();
//!
//! // DSP features (from Phase 2 DSP pipeline)
//! let features = [
//!     0.5,   // divergence_norm
//!     0.0,   // alpha_norm
//!     1.5,   // fractal_dim
//!     0.75,  // hurst
//!     1.0,   // regime
//!     1.0,   // divergence_sign
//!     0.0,   // alpha_deviation
//!     0.3,   // regime_confidence
//! ];
//!
//! // Model prediction (would come from neural network)
//! let signal = TradingSignal::new(0.8, 0.15, 0.05);
//!
//! // Evaluate axioms
//! let results = axioms.evaluate_all(&features, &signal);
//!
//! // Compute semantic loss
//! let semantic_loss = axioms.compute_semantic_loss(&results);
//!
//! println!("Semantic loss: {:.4}", semantic_loss);
//! ```
//!
//! # Module Organization
//!
//! - [`fuzzy_ops`]: Differentiable fuzzy logic operators (T-norms, implications)
//! - [`predicates`]: Market-specific fuzzy predicates
//! - [`axioms`]: Complete axiom library (10 trading rules)
//! - [`config`]: Configuration structures and builders
//!
//! # Key Concepts
//!
//! ## Fuzzy Logic
//!
//! Traditional logic uses binary truth values (true/false). Fuzzy logic
//! extends this to continuous truth values in [0, 1]:
//!
//! - **T-norms** (AND): `T(a, b) = a × b` (Product T-norm)
//! - **Implications** (IF-THEN): `I(a, b) = 1 - a + a×b` (Reichenbach)
//! - **Negation** (NOT): `¬a = 1 - a`
//!
//! All operators are differentiable, enabling gradient-based learning.
//!
//! ## Logic Tensor Networks
//!
//! LTNs ground logical predicates as neural networks:
//!
//! 1. **Predicates** map features to fuzzy truth values
//! 2. **Axioms** combine predicates into logical rules
//! 3. **Satisfaction** measures how well axioms are satisfied
//! 4. **Semantic Loss** guides learning toward axiom satisfaction
//!
//! ## Hybrid Loss
//!
//! The model is trained with a combination of:
//!
//! ```text
//! L_total = α × L_supervised + (1-α) × L_semantic
//! ```
//!
//! Where:
//! - **L_supervised**: Cross-entropy on labeled data (learn from examples)
//! - **L_semantic**: Negative axiom satisfaction (learn from rules)
//! - **α ∈ [0,1]**: Balance between data and knowledge
//!
//! ## Market Axioms
//!
//! The system encodes 10 core trading rules:
//!
//! 1. **Trending + Positive Div → Long**: Follow momentum
//! 2. **Trending + Negative Div → Short**: Follow momentum
//! 3. **Mean-Reverting + Positive Div → Short**: Fade overbought
//! 4. **Mean-Reverting + Negative Div → Long**: Fade oversold
//! 5. **Low Confidence → Neutral**: Risk control (CRITICAL)
//! 6. **High Noise → Caution**: Reduce exposure
//! 7. **Contradiction → Neutral**: Logical consistency
//! 8. **Extreme Alpha → Neutral**: Volatility caution
//! 9. **Probability Sum = 1**: Mathematical constraint
//! 10. **Confidence Monotonicity**: Preference for consistency
//!
//! # Performance Requirements
//!
//! - **Inference Latency**: <10μs median, <50μs P99
//! - **Total Pipeline**: <20μs (DSP + LTN + risk)
//! - **Throughput**: >100K inferences/sec
//! - **Memory**: <100KB per model instance
//!
//! # Training Strategy
//!
//! 1. **Synthetic Data**: Generate trending/MR/random markets
//! 2. **Historical Data**: Real market tick data
//! 3. **Walk-Forward Validation**: Rolling window evaluation
//! 4. **Semantic Weight Schedule**:
//!    - Early: α = 0.8 (focus on data)
//!    - Mid: α = 0.5 (balance)
//!    - Late: α = 0.3 (focus on axioms)
//!
//! # References
//!
//! - Serafini, L., & Garcez, A. (2016). "Logic Tensor Networks". arXiv:1606.04422
//! - Badreddine, S., et al. (2022). "Logic Tensor Networks". AI, 303, 103649
//! - Klement, E. P., et al. (2000). "Triangular Norms". Springer
//!
//! # Examples
//!
//! ## Evaluating Individual Axioms
//!
//! ```rust,ignore
//! use janus_ltn::axioms::{axiom_trending_long, axiom_low_confidence_neutral};
//! use ltn::predicates::TradingSignal;
//!
//! let features = [1.5, 0.0, 1.5, 0.75, 1.0, 1.0, 0.0, 0.4];
//! let signal = TradingSignal::new(0.9, 0.08, 0.02);
//!
//! let satisfaction = axiom_trending_long(&features, &signal);
//! println!("Axiom 1 satisfaction: {:.3}", satisfaction);
//! ```
//!
//! ## Custom Axiom Weights
//!
//! ```rust,ignore
//! use janus_ltn::axioms::AxiomLibrary;
//!
//! // Emphasize risk management
//! let weights = [
//!     1.0, 1.0,  // Trading axioms (lower weight)
//!     1.0, 1.0,
//!     5.0,       // Low confidence → neutral (CRITICAL)
//!     4.0,       // High noise → caution
//!     3.0,       // Contradiction → neutral
//!     2.0,       // Extreme alpha → neutral
//!     5.0,       // Probability sum
//!     1.0,       // Confidence monotonicity
//! ];
//!
//! let axioms = AxiomLibrary::new(weights);
//! ```
//!
//! ## Fuzzy Predicate Composition
//!
//! ```rust,ignore
//! use janus_ltn::predicates::{is_trending, divergence_positive, high_regime_confidence};
//! use ltn::fuzzy_ops::product_tnorm;
//!
//! let features = [1.5, 0.0, 1.5, 0.75, 1.0, 1.0, 0.0, 0.4];
//!
//! // Composite predicate: trending AND positive divergence AND high confidence
//! let trending = is_trending(&features);
//! let div_pos = divergence_positive(&features);
//! let high_conf = high_regime_confidence(&features);
//!
//! let favorable_long = product_tnorm(
//!     product_tnorm(trending, div_pos),
//!     high_conf
//! );
//!
//! println!("Long favorable: {:.3}", favorable_long);
//! ```

// Re-export public API
pub mod axioms;
pub mod config;
pub mod fuzzy_ops;
pub mod predicates;

// Convenience re-exports
pub use axioms::{AxiomLibrary, AxiomResult, AxiomStats};
pub use config::{AxiomConfig, InferenceConfig, LtnConfig, ModelConfig, TrainingConfig};
pub use predicates::{Action, TradingSignal};

/// LTN module version
pub const VERSION: &str = "0.1.0";

/// Number of input features (from DSP pipeline)
pub const INPUT_DIM: usize = 8;

/// Number of output classes (long, neutral, short)
pub const OUTPUT_DIM: usize = 3;

/// Number of core axioms
pub const NUM_AXIOMS: usize = 10;

/// Default semantic weight (α in hybrid loss)
pub const DEFAULT_SEMANTIC_WEIGHT: f64 = 0.5;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_constants() {
        assert_eq!(INPUT_DIM, 8);
        assert_eq!(OUTPUT_DIM, 3);
        assert_eq!(NUM_AXIOMS, 10);
    }

    #[test]
    fn test_public_api() {
        // Verify all public types are accessible
        let config = LtnConfig::default();
        let axioms = AxiomLibrary::default();
        let signal = TradingSignal::new(0.7, 0.2, 0.1);

        assert_eq!(config.model.input_dim, INPUT_DIM);
        assert_eq!(axioms.weights.len(), NUM_AXIOMS);
        assert_eq!(signal.predicted_action(), Action::Long);
    }

    #[test]
    fn test_end_to_end_evaluation() {
        // Full pipeline: features → axioms → loss
        let features = [1.5, 0.0, 1.5, 0.75, 1.0, 1.0, 0.0, 0.4];
        let signal = TradingSignal::new(0.8, 0.15, 0.05);

        let axioms = AxiomLibrary::default();
        let results = axioms.evaluate_all(&features, &signal);
        let loss = axioms.compute_semantic_loss(&results);

        assert_eq!(results.len(), NUM_AXIOMS);
        assert!(loss < 0.0); // Negative (we maximize satisfaction)

        // All results should have valid satisfaction
        for result in &results {
            assert!(result.satisfaction >= 0.0 && result.satisfaction <= 1.0);
        }
    }

    #[test]
    fn test_fuzzy_logic_integration() {
        use fuzzy_ops::{product_tnorm, reichenbach_implication};
        use predicates::{divergence_positive, is_trending};

        let features = [1.5, 0.0, 1.5, 0.75, 1.0, 1.0, 0.0, 0.4];

        let trending = is_trending(&features);
        let div_pos = divergence_positive(&features);

        let premise = product_tnorm(trending, div_pos);

        // Simulate conclusion (should_long)
        let conclusion = 0.9;

        let satisfaction = reichenbach_implication(premise, conclusion);

        assert!(satisfaction > 0.7);
    }
}
