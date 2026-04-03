//! Logical Predicates for Logic Tensor Networks.
//!
//! This module provides learnable and pre-defined predicates that map neural network
//! outputs to truth values in [0, 1]. Predicates form the building blocks of logical
//! formulas in neuro-symbolic reasoning.
//!
//! # Predicate Types
//!
//! 1. **Learnable Predicates**: Neural network-based predicates with trainable parameters
//! 2. **Threshold Predicates**: Simple comparison-based predicates (>, <, in range)
//! 3. **Similarity Predicates**: Distance-based predicates for embeddings
//! 4. **Composite Predicates**: Combinations of other predicates
//!
//! # Example
//!
//! ```ignore
//! use logic::predicates::{LearnablePredicate, ThresholdPredicate, PredicateConfig};
//! use candle_core::{Device, Tensor};
//!
//! let device = Device::Cpu;
//!
//! // Create a learnable predicate
//! let pred = LearnablePredicate::new(PredicateConfig::default(), &device)?;
//! let input = Tensor::randn(0.0, 1.0, (4, 128), &device)?;
//! let truth_values = pred.forward(&input)?; // (4,) tensor with values in [0, 1]
//!
//! // Create a threshold predicate
//! let threshold_pred = ThresholdPredicate::greater_than(0.5);
//! let values = Tensor::new(&[0.3f32, 0.7, 0.5, 0.9], &device)?;
//! let truth = threshold_pred.evaluate(&values)?;
//! ```

use candle_core::{D, DType, Device, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder, VarMap, linear};
use serde::{Deserialize, Serialize};

// =============================================================================
// Predicate Trait
// =============================================================================

/// Trait for all predicates that can evaluate truth values.
pub trait Predicate {
    /// Evaluate the predicate on input tensor.
    ///
    /// # Arguments
    /// * `input` - Input tensor to evaluate
    ///
    /// # Returns
    /// Tensor of truth values in [0, 1], typically shape (batch,) or (batch, 1)
    fn evaluate(&self, input: &Tensor) -> Result<Tensor>;

    /// Get the name of this predicate.
    fn name(&self) -> &str;
}

// =============================================================================
// Learnable Predicate
// =============================================================================

/// Configuration for learnable predicates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnablePredicateConfig {
    /// Input dimension (e.g., embedding size from ViViT)
    pub input_dim: usize,

    /// Hidden dimension for the predicate network
    pub hidden_dim: usize,

    /// Number of hidden layers
    pub num_layers: usize,

    /// Dropout probability
    pub dropout: f32,

    /// Name of the predicate
    pub name: String,
}

impl Default for LearnablePredicateConfig {
    fn default() -> Self {
        Self {
            input_dim: 256,
            hidden_dim: 64,
            num_layers: 2,
            dropout: 0.1,
            name: "learnable_predicate".to_string(),
        }
    }
}

impl LearnablePredicateConfig {
    /// Create config for a simple single-layer predicate.
    pub fn simple(input_dim: usize, name: &str) -> Self {
        Self {
            input_dim,
            hidden_dim: input_dim / 2,
            num_layers: 1,
            dropout: 0.0,
            name: name.to_string(),
        }
    }

    /// Create config for a deep predicate network.
    pub fn deep(input_dim: usize, hidden_dim: usize, num_layers: usize, name: &str) -> Self {
        Self {
            input_dim,
            hidden_dim,
            num_layers,
            dropout: 0.1,
            name: name.to_string(),
        }
    }
}

/// A learnable predicate implemented as a small neural network.
///
/// Maps high-dimensional inputs (e.g., embeddings) to truth values [0, 1].
/// The network architecture is: input → [hidden → ReLU] × n → output → sigmoid
pub struct LearnablePredicate {
    layers: Vec<Linear>,
    output: Linear,
    config: LearnablePredicateConfig,
    #[allow(dead_code)]
    device: Device,
}

impl LearnablePredicate {
    /// Create a new learnable predicate.
    ///
    /// # Arguments
    /// * `config` - Configuration for the predicate
    /// * `vb` - Variable builder for creating learnable parameters
    ///
    /// # Returns
    /// A new LearnablePredicate instance
    pub fn new(config: LearnablePredicateConfig, vb: VarBuilder) -> Result<Self> {
        let mut layers = Vec::new();
        let mut in_dim = config.input_dim;

        for i in 0..config.num_layers {
            let layer = linear(in_dim, config.hidden_dim, vb.pp(format!("layer_{i}")))?;
            layers.push(layer);
            in_dim = config.hidden_dim;
        }

        let output = linear(in_dim, 1, vb.pp("output"))?;

        Ok(Self {
            layers,
            output,
            device: vb.device().clone(),
            config,
        })
    }

    /// Create a predicate with randomly initialized weights.
    pub fn new_random(config: LearnablePredicateConfig, device: &Device) -> Result<(Self, VarMap)> {
        let var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, device);
        let pred = Self::new(config, vb)?;
        Ok((pred, var_map))
    }

    /// Forward pass through the predicate network.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut x = input.clone();

        for layer in &self.layers {
            x = layer.forward(&x)?;
            x = x.relu()?;
        }

        let logits = self.output.forward(&x)?;
        let squeezed = logits.squeeze(D::Minus1)?;

        // Apply sigmoid to get truth value in [0, 1]
        sigmoid(&squeezed)
    }

    /// Get the configuration.
    pub fn config(&self) -> &LearnablePredicateConfig {
        &self.config
    }
}

impl Predicate for LearnablePredicate {
    fn evaluate(&self, input: &Tensor) -> Result<Tensor> {
        self.forward(input)
    }

    fn name(&self) -> &str {
        &self.config.name
    }
}

// =============================================================================
// Threshold Predicates
// =============================================================================

/// Type of threshold comparison.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ThresholdType {
    /// Value > threshold
    GreaterThan,
    /// Value < threshold
    LessThan,
    /// Value >= threshold
    GreaterOrEqual,
    /// Value <= threshold
    LessOrEqual,
    /// value in [lower, upper]
    InRange,
}

/// Configuration for threshold predicates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdPredicateConfig {
    /// Type of threshold comparison
    pub threshold_type: ThresholdType,

    /// Primary threshold value
    pub threshold: f32,

    /// Upper bound (for InRange type)
    pub upper_bound: Option<f32>,

    /// Steepness of the sigmoid approximation (higher = sharper transition)
    pub steepness: f32,

    /// Name of the predicate
    pub name: String,
}

impl Default for ThresholdPredicateConfig {
    fn default() -> Self {
        Self {
            threshold_type: ThresholdType::GreaterThan,
            threshold: 0.5,
            upper_bound: None,
            steepness: 10.0,
            name: "threshold_predicate".to_string(),
        }
    }
}

/// A threshold-based predicate using smooth sigmoid approximations.
///
/// Converts hard threshold comparisons into differentiable truth values.
#[derive(Debug, Clone)]
pub struct ThresholdPredicate {
    config: ThresholdPredicateConfig,
}

impl ThresholdPredicate {
    /// Create a new threshold predicate with the given configuration.
    pub fn new(config: ThresholdPredicateConfig) -> Self {
        Self { config }
    }

    /// Create a "greater than" predicate: value > threshold
    pub fn greater_than(threshold: f32) -> Self {
        Self::new(ThresholdPredicateConfig {
            threshold_type: ThresholdType::GreaterThan,
            threshold,
            name: format!("gt_{threshold}"),
            ..Default::default()
        })
    }

    /// Create a "less than" predicate: value < threshold
    pub fn less_than(threshold: f32) -> Self {
        Self::new(ThresholdPredicateConfig {
            threshold_type: ThresholdType::LessThan,
            threshold,
            name: format!("lt_{threshold}"),
            ..Default::default()
        })
    }

    /// Create a "greater or equal" predicate: value >= threshold
    pub fn greater_or_equal(threshold: f32) -> Self {
        Self::new(ThresholdPredicateConfig {
            threshold_type: ThresholdType::GreaterOrEqual,
            threshold,
            name: format!("gte_{threshold}"),
            ..Default::default()
        })
    }

    /// Create a "less or equal" predicate: value <= threshold
    pub fn less_or_equal(threshold: f32) -> Self {
        Self::new(ThresholdPredicateConfig {
            threshold_type: ThresholdType::LessOrEqual,
            threshold,
            name: format!("lte_{threshold}"),
            ..Default::default()
        })
    }

    /// Create an "in range" predicate: lower <= value <= upper
    pub fn in_range(lower: f32, upper: f32) -> Self {
        Self::new(ThresholdPredicateConfig {
            threshold_type: ThresholdType::InRange,
            threshold: lower,
            upper_bound: Some(upper),
            name: format!("in_range_{lower}_{upper}"),
            ..Default::default()
        })
    }

    /// Create with custom steepness (controls sharpness of transition).
    pub fn with_steepness(mut self, steepness: f32) -> Self {
        self.config.steepness = steepness;
        self
    }

    /// Evaluate the threshold predicate.
    pub fn evaluate_threshold(&self, input: &Tensor) -> Result<Tensor> {
        let k = self.config.steepness;
        let t = self.config.threshold;

        match self.config.threshold_type {
            ThresholdType::GreaterThan | ThresholdType::GreaterOrEqual => {
                // sigmoid(k * (x - t))
                // Using affine: k * (x - t) = k*x - k*t
                let scaled = input.affine(k as f64, -(k * t) as f64)?;
                sigmoid(&scaled)
            }
            ThresholdType::LessThan | ThresholdType::LessOrEqual => {
                // sigmoid(k * (t - x))
                // Using affine: k * (t - x) = -k*x + k*t
                let scaled = input.affine(-(k as f64), (k * t) as f64)?;
                sigmoid(&scaled)
            }
            ThresholdType::InRange => {
                let upper = self.config.upper_bound.unwrap_or(t + 1.0);

                // sigmoid(k * (x - lower)) * sigmoid(k * (upper - x))
                // lower_scaled = k*x - k*lower
                let lower_scaled = input.affine(k as f64, -(k * t) as f64)?;
                let lower_sig = sigmoid(&lower_scaled)?;

                // upper_scaled = -k*x + k*upper
                let upper_scaled = input.affine(-(k as f64), (k * upper) as f64)?;
                let upper_sig = sigmoid(&upper_scaled)?;

                lower_sig * upper_sig
            }
        }
    }
}

impl Predicate for ThresholdPredicate {
    fn evaluate(&self, input: &Tensor) -> Result<Tensor> {
        self.evaluate_threshold(input)
    }

    fn name(&self) -> &str {
        &self.config.name
    }
}

// =============================================================================
// Similarity Predicate
// =============================================================================

/// Type of similarity metric.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SimilarityType {
    /// Cosine similarity
    Cosine,
    /// Euclidean distance (converted to similarity)
    Euclidean,
    /// Dot product
    DotProduct,
}

/// Configuration for similarity predicates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityPredicateConfig {
    /// Type of similarity metric
    pub similarity_type: SimilarityType,

    /// Temperature for scaling similarity scores
    pub temperature: f32,

    /// Name of the predicate
    pub name: String,
}

impl Default for SimilarityPredicateConfig {
    fn default() -> Self {
        Self {
            similarity_type: SimilarityType::Cosine,
            temperature: 1.0,
            name: "similarity_predicate".to_string(),
        }
    }
}

/// A similarity-based predicate for comparing embeddings.
///
/// Computes similarity between input embeddings and a reference embedding,
/// returning a truth value indicating how similar they are.
pub struct SimilarityPredicate {
    reference: Tensor,
    config: SimilarityPredicateConfig,
}

impl SimilarityPredicate {
    /// Create a new similarity predicate with a reference embedding.
    ///
    /// # Arguments
    /// * `reference` - Reference embedding to compare against
    /// * `config` - Configuration for the predicate
    pub fn new(reference: Tensor, config: SimilarityPredicateConfig) -> Self {
        Self { reference, config }
    }

    /// Create a cosine similarity predicate.
    pub fn cosine(reference: Tensor) -> Self {
        Self::new(
            reference,
            SimilarityPredicateConfig {
                similarity_type: SimilarityType::Cosine,
                ..Default::default()
            },
        )
    }

    /// Create a Euclidean distance-based similarity predicate.
    pub fn euclidean(reference: Tensor) -> Self {
        Self::new(
            reference,
            SimilarityPredicateConfig {
                similarity_type: SimilarityType::Euclidean,
                ..Default::default()
            },
        )
    }

    /// Compute similarity between input and reference.
    pub fn compute_similarity(&self, input: &Tensor) -> Result<Tensor> {
        match self.config.similarity_type {
            SimilarityType::Cosine => self.cosine_similarity(input),
            SimilarityType::Euclidean => self.euclidean_similarity(input),
            SimilarityType::DotProduct => self.dot_product_similarity(input),
        }
    }

    fn cosine_similarity(&self, input: &Tensor) -> Result<Tensor> {
        // Normalize both vectors
        let input_norm = normalize_l2(input)?;
        let ref_norm = normalize_l2(&self.reference)?;

        // Compute dot product
        let dot = (&input_norm * &ref_norm)?.sum(D::Minus1)?;

        // Scale by temperature and apply sigmoid to get [0, 1]
        // Cosine similarity is already in [-1, 1], so we map to [0, 1]
        // (temp * dot + 1) / 2 = 0.5 * temp * dot + 0.5
        let temp = self.config.temperature;
        dot.affine((temp * 0.5) as f64, 0.5)
    }

    fn euclidean_similarity(&self, input: &Tensor) -> Result<Tensor> {
        // Compute L2 distance
        let diff = (input - &self.reference)?;
        let sq_diff = (&diff * &diff)?;
        let sum_sq = sq_diff.sum(D::Minus1)?;
        let distance = sum_sq.sqrt()?;

        // Convert distance to similarity: exp(-distance / temperature)
        let neg_scaled = distance.affine(-1.0 / self.config.temperature as f64, 0.0)?;
        neg_scaled.exp()
    }

    fn dot_product_similarity(&self, input: &Tensor) -> Result<Tensor> {
        let dot = (input * &self.reference)?.sum(D::Minus1)?;
        let scaled = dot.affine(self.config.temperature as f64, 0.0)?;
        sigmoid(&scaled)
    }
}

impl Predicate for SimilarityPredicate {
    fn evaluate(&self, input: &Tensor) -> Result<Tensor> {
        self.compute_similarity(input)
    }

    fn name(&self) -> &str {
        &self.config.name
    }
}

// =============================================================================
// Pre-defined Trading Predicates
// =============================================================================

/// Pre-defined predicates for trading/risk constraints.
pub struct TradingPredicates;

impl TradingPredicates {
    /// Predicate: position size is within acceptable bounds (0, max_size)
    pub fn position_size_valid(max_size: f32) -> ThresholdPredicate {
        ThresholdPredicate::in_range(0.0, max_size).with_steepness(5.0)
    }

    /// Predicate: confidence is high enough for action
    pub fn high_confidence(threshold: f32) -> ThresholdPredicate {
        ThresholdPredicate::greater_than(threshold).with_steepness(15.0)
    }

    /// Predicate: risk is acceptable (below threshold)
    pub fn acceptable_risk(max_risk: f32) -> ThresholdPredicate {
        ThresholdPredicate::less_than(max_risk).with_steepness(10.0)
    }

    /// Predicate: signal is strong enough to act on
    pub fn strong_signal(threshold: f32) -> ThresholdPredicate {
        ThresholdPredicate::greater_than(threshold).with_steepness(20.0)
    }

    /// Predicate: volatility is in acceptable range
    pub fn volatility_acceptable(min_vol: f32, max_vol: f32) -> ThresholdPredicate {
        ThresholdPredicate::in_range(min_vol, max_vol).with_steepness(8.0)
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Sigmoid activation function.
fn sigmoid(x: &Tensor) -> Result<Tensor> {
    let neg_x = x.neg()?;
    let exp_neg = neg_x.exp()?;
    let one = Tensor::ones_like(&exp_neg)?;
    let denom = (&one + &exp_neg)?;
    one.broadcast_div(&denom)
}

/// L2 normalize a tensor along the last dimension.
fn normalize_l2(x: &Tensor) -> Result<Tensor> {
    let sq = (x * x)?;
    let sum_sq = sq.sum_keepdim(D::Minus1)?;
    let norm = (sum_sq + 1e-8)?.sqrt()?;
    x.broadcast_div(&norm)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_learnable_predicate_creation() {
        let device = Device::Cpu;
        let config = LearnablePredicateConfig::simple(128, "test_pred");
        let (pred, _var_map) = LearnablePredicate::new_random(config, &device).unwrap();

        assert_eq!(pred.name(), "test_pred");
    }

    #[test]
    fn test_learnable_predicate_forward() {
        let device = Device::Cpu;
        let config = LearnablePredicateConfig {
            input_dim: 64,
            hidden_dim: 32,
            num_layers: 2,
            dropout: 0.0,
            name: "test".to_string(),
        };
        let (pred, _) = LearnablePredicate::new_random(config, &device).unwrap();

        let input = Tensor::randn(0.0f32, 1.0, (4, 64), &device).unwrap();
        let output = pred.forward(&input).unwrap();

        assert_eq!(output.dims(), &[4]);

        // Check output is in [0, 1]
        let values: Vec<f32> = output.to_vec1().unwrap();
        for v in values {
            assert!(v >= 0.0 && v <= 1.0, "Output {v} out of bounds");
        }
    }

    #[test]
    fn test_threshold_greater_than() {
        let device = Device::Cpu;
        let pred = ThresholdPredicate::greater_than(0.5);

        let input = Tensor::new(&[0.3f32, 0.5, 0.7, 0.9], &device).unwrap();
        let output = pred.evaluate(&input).unwrap();
        let values: Vec<f32> = output.to_vec1().unwrap();

        // 0.3 < 0.5 → low truth value
        assert!(values[0] < 0.5);
        // 0.7 > 0.5 → high truth value
        assert!(values[2] > 0.5);
        // 0.9 >> 0.5 → very high truth value
        assert!(values[3] > 0.9);
    }

    #[test]
    fn test_threshold_less_than() {
        let device = Device::Cpu;
        let pred = ThresholdPredicate::less_than(0.5);

        let input = Tensor::new(&[0.3f32, 0.5, 0.7, 0.9], &device).unwrap();
        let output = pred.evaluate(&input).unwrap();
        let values: Vec<f32> = output.to_vec1().unwrap();

        // 0.3 < 0.5 → high truth value
        assert!(values[0] > 0.5);
        // 0.7 > 0.5 → low truth value
        assert!(values[2] < 0.5);
    }

    #[test]
    fn test_threshold_in_range() {
        let device = Device::Cpu;
        let pred = ThresholdPredicate::in_range(0.3, 0.7);

        let input = Tensor::new(&[0.1f32, 0.5, 0.9], &device).unwrap();
        let output = pred.evaluate(&input).unwrap();
        let values: Vec<f32> = output.to_vec1().unwrap();

        // 0.1 outside range → low
        assert!(values[0] < 0.5);
        // 0.5 inside range → high
        assert!(values[1] > 0.5);
        // 0.9 outside range → low
        assert!(values[2] < 0.5);
    }

    #[test]
    fn test_similarity_cosine() {
        let device = Device::Cpu;
        let reference = Tensor::new(&[1.0f32, 0.0, 0.0, 0.0], &device).unwrap();
        let pred = SimilarityPredicate::cosine(reference);

        // Same direction → high similarity
        let same = Tensor::new(&[1.0f32, 0.0, 0.0, 0.0], &device).unwrap();
        let same_result = pred.evaluate(&same).unwrap();
        let same_val: f32 = same_result.to_scalar().unwrap();
        assert!(same_val > 0.9, "Same direction should have high similarity");

        // Orthogonal → medium similarity (cosine = 0 → mapped to 0.5)
        let ortho = Tensor::new(&[0.0f32, 1.0, 0.0, 0.0], &device).unwrap();
        let ortho_result = pred.evaluate(&ortho).unwrap();
        let ortho_val: f32 = ortho_result.to_scalar().unwrap();
        assert!(
            (ortho_val - 0.5).abs() < 0.1,
            "Orthogonal should have ~0.5 similarity"
        );

        // Opposite → low similarity
        let opposite = Tensor::new(&[-1.0f32, 0.0, 0.0, 0.0], &device).unwrap();
        let opp_result = pred.evaluate(&opposite).unwrap();
        let opp_val: f32 = opp_result.to_scalar().unwrap();
        assert!(
            opp_val < 0.1,
            "Opposite direction should have low similarity"
        );
    }

    #[test]
    fn test_similarity_euclidean() {
        let device = Device::Cpu;
        let reference = Tensor::new(&[0.0f32, 0.0, 0.0, 0.0], &device).unwrap();
        let pred = SimilarityPredicate::euclidean(reference);

        // Zero distance → high similarity
        let same = Tensor::new(&[0.0f32, 0.0, 0.0, 0.0], &device).unwrap();
        let same_result = pred.evaluate(&same).unwrap();
        let same_val: f32 = same_result.to_scalar().unwrap();
        assert!(same_val > 0.99, "Zero distance should have ~1.0 similarity");

        // Large distance → low similarity
        let far = Tensor::new(&[10.0f32, 10.0, 10.0, 10.0], &device).unwrap();
        let far_result = pred.evaluate(&far).unwrap();
        let far_val: f32 = far_result.to_scalar().unwrap();
        assert!(far_val < 0.1, "Large distance should have low similarity");
    }

    #[test]
    fn test_trading_predicates() {
        let device = Device::Cpu;

        // Test position size validity
        let pos_pred = TradingPredicates::position_size_valid(100.0);
        let sizes = Tensor::new(&[50.0f32, 150.0, -10.0], &device).unwrap();
        let result = pos_pred.evaluate(&sizes).unwrap();
        let values: Vec<f32> = result.to_vec1().unwrap();

        assert!(values[0] > 0.5, "50 should be valid (< 100)");
        assert!(values[1] < 0.5, "150 should be invalid (> 100)");
        assert!(values[2] < 0.5, "-10 should be invalid (< 0)");

        // Test high confidence
        let conf_pred = TradingPredicates::high_confidence(0.8);
        let confs = Tensor::new(&[0.5f32, 0.8, 0.95], &device).unwrap();
        let result = conf_pred.evaluate(&confs).unwrap();
        let values: Vec<f32> = result.to_vec1().unwrap();

        assert!(values[0] < 0.5, "0.5 confidence should be low");
        assert!(values[2] > 0.8, "0.95 confidence should be high");
    }

    #[test]
    fn test_predicate_trait() {
        let device = Device::Cpu;
        let pred: Box<dyn Predicate> = Box::new(ThresholdPredicate::greater_than(0.5));

        let input = Tensor::new(&[0.3f32, 0.7], &device).unwrap();
        let output = pred.evaluate(&input).unwrap();

        assert_eq!(output.dims(), &[2]);
        assert_eq!(pred.name(), "gt_0.5");
    }

    #[test]
    fn test_steepness_affects_sharpness() {
        let device = Device::Cpu;
        let input = Tensor::new(&[0.49f32, 0.51], &device).unwrap();

        // Low steepness → gradual transition
        let soft_pred = ThresholdPredicate::greater_than(0.5).with_steepness(1.0);
        let soft_output = soft_pred.evaluate(&input).unwrap();
        let soft_values: Vec<f32> = soft_output.to_vec1().unwrap();
        let soft_diff = (soft_values[1] - soft_values[0]).abs();

        // High steepness → sharp transition
        let sharp_pred = ThresholdPredicate::greater_than(0.5).with_steepness(100.0);
        let sharp_output = sharp_pred.evaluate(&input).unwrap();
        let sharp_values: Vec<f32> = sharp_output.to_vec1().unwrap();
        let sharp_diff = (sharp_values[1] - sharp_values[0]).abs();

        assert!(
            sharp_diff > soft_diff,
            "Sharp predicate should have larger difference"
        );
    }

    #[test]
    fn test_batched_predicate_evaluation() {
        let device = Device::Cpu;
        let pred = ThresholdPredicate::greater_than(0.5);

        // 2D input: batch of 3, 4 values each
        let input = Tensor::new(
            &[
                [0.3f32, 0.7, 0.2, 0.9],
                [0.6, 0.4, 0.8, 0.1],
                [0.5, 0.5, 0.5, 0.5],
            ],
            &device,
        )
        .unwrap();

        let output = pred.evaluate(&input).unwrap();
        assert_eq!(output.dims(), &[3, 4]);
    }
}
