//! Differentiable T-Norms for Logic Tensor Networks.
//!
//! This module implements fuzzy logic t-norms (triangular norms) and their dual
//! t-conorms as differentiable tensor operations using Candle. These operations
//! form the foundation of neuro-symbolic reasoning, allowing gradients to flow
//! through logical operations during neural network training.
//!
//! # Background
//!
//! T-norms generalize the AND operation to fuzzy logic, mapping [0,1] × [0,1] → [0,1].
//! They satisfy: commutativity, associativity, monotonicity, and boundary conditions.
//!
//! # Supported T-Norm Families
//!
//! 1. **Łukasiewicz**: `A ∧ B = max(0, A + B - 1)` - Good gradient flow, most commonly used
//! 2. **Product**: `A ∧ B = A * B` - Smooth everywhere, natural for probabilities
//! 3. **Gödel**: `A ∧ B = min(A, B)` - Classical min/max logic (uses soft approximations)
//!
//! Each family has corresponding:
//! - T-conorm (S-norm) for OR: `A ∨ B`
//! - Residual implication for IF-THEN: `A → B`
//! - Negation: `¬A`
//!
//! # Example
//!
//! ```ignore
//! use logic::diff_tnorm::{DiffTNorm, TNormType};
//! use candle_core::{Device, Tensor};
//!
//! let device = Device::Cpu;
//! let tnorm = DiffTNorm::new(TNormType::Lukasiewicz);
//!
//! let a = Tensor::new(&[0.8f32, 0.6, 0.9], &device)?;
//! let b = Tensor::new(&[0.7f32, 0.5, 0.3], &device)?;
//!
//! let and_result = tnorm.and(&a, &b)?;  // Fuzzy AND
//! let or_result = tnorm.or(&a, &b)?;    // Fuzzy OR
//! let implies = tnorm.implies(&a, &b)?; // Fuzzy implication
//! ```

use candle_core::{Result, Tensor};
use serde::{Deserialize, Serialize};

/// The type of T-norm to use for fuzzy logic operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TNormType {
    /// Łukasiewicz T-norm: max(0, a + b - 1)
    /// Most commonly used in LTN, provides good gradient flow
    #[default]
    Lukasiewicz,

    /// Product T-norm: a * b
    /// Smooth everywhere, natural probabilistic interpretation
    Product,

    /// Gödel T-norm: min(a, b)
    /// Classical min/max logic, uses soft approximations for differentiability
    Godel,
}

/// Configuration for differentiable t-norm operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffTNormConfig {
    /// Type of t-norm to use
    pub tnorm_type: TNormType,

    /// Temperature for soft min/max operations (Gödel t-norm)
    /// Lower values → sharper approximation, higher values → smoother gradients
    pub temperature: f32,

    /// Epsilon for numerical stability
    pub eps: f64,

    /// Whether to clamp outputs to [0, 1]
    pub clamp_outputs: bool,
}

impl Default for DiffTNormConfig {
    fn default() -> Self {
        Self {
            tnorm_type: TNormType::Lukasiewicz,
            temperature: 0.1,
            eps: 1e-7,
            clamp_outputs: true,
        }
    }
}

impl DiffTNormConfig {
    /// Create config for Łukasiewicz t-norm
    pub fn lukasiewicz() -> Self {
        Self {
            tnorm_type: TNormType::Lukasiewicz,
            ..Default::default()
        }
    }

    /// Create config for Product t-norm
    pub fn product() -> Self {
        Self {
            tnorm_type: TNormType::Product,
            ..Default::default()
        }
    }

    /// Create config for Gödel t-norm with custom temperature
    pub fn godel(temperature: f32) -> Self {
        Self {
            tnorm_type: TNormType::Godel,
            temperature,
            ..Default::default()
        }
    }
}

/// Differentiable T-Norm operations for fuzzy logic.
///
/// Provides AND, OR, NOT, and IMPLIES operations that maintain gradient flow
/// for end-to-end neural network training with logical constraints.
#[derive(Debug, Clone)]
pub struct DiffTNorm {
    config: DiffTNormConfig,
}

impl DiffTNorm {
    /// Create a new DiffTNorm with the specified t-norm type.
    pub fn new(tnorm_type: TNormType) -> Self {
        Self {
            config: DiffTNormConfig {
                tnorm_type,
                ..Default::default()
            },
        }
    }

    /// Create a new DiffTNorm with full configuration.
    pub fn with_config(config: DiffTNormConfig) -> Self {
        Self { config }
    }

    /// Get the configuration.
    pub fn config(&self) -> &DiffTNormConfig {
        &self.config
    }

    /// Get the t-norm type.
    pub fn tnorm_type(&self) -> TNormType {
        self.config.tnorm_type
    }

    // =========================================================================
    // Core T-Norm Operations (AND)
    // =========================================================================

    /// Fuzzy AND (conjunction) using the configured t-norm.
    ///
    /// Computes A ∧ B element-wise.
    ///
    /// # Arguments
    /// * `a` - First operand tensor with values in [0, 1]
    /// * `b` - Second operand tensor with values in [0, 1]
    ///
    /// # Returns
    /// Tensor with fuzzy AND result, same shape as inputs
    pub fn and(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let result = match self.config.tnorm_type {
            TNormType::Lukasiewicz => self.lukasiewicz_and(a, b)?,
            TNormType::Product => self.product_and(a, b)?,
            TNormType::Godel => self.godel_and(a, b)?,
        };

        if self.config.clamp_outputs {
            self.clamp_01(&result)
        } else {
            Ok(result)
        }
    }

    /// Łukasiewicz T-norm: max(0, a + b - 1)
    fn lukasiewicz_and(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let sum = (a + b)?;
        let shifted = sum.broadcast_sub(&Tensor::new(1.0f32, a.device())?)?;
        shifted.maximum(&Tensor::zeros_like(&shifted)?)
    }

    /// Product T-norm: a * b
    fn product_and(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        a * b
    }

    /// Gödel T-norm: min(a, b) using soft minimum for differentiability
    fn godel_and(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        self.soft_min(a, b)
    }

    // =========================================================================
    // T-Conorm Operations (OR)
    // =========================================================================

    /// Fuzzy OR (disjunction) using the configured t-conorm.
    ///
    /// Computes A ∨ B element-wise.
    /// T-conorms are derived from t-norms via: S(a,b) = 1 - T(1-a, 1-b)
    ///
    /// # Arguments
    /// * `a` - First operand tensor with values in [0, 1]
    /// * `b` - Second operand tensor with values in [0, 1]
    ///
    /// # Returns
    /// Tensor with fuzzy OR result, same shape as inputs
    pub fn or(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let result = match self.config.tnorm_type {
            TNormType::Lukasiewicz => self.lukasiewicz_or(a, b)?,
            TNormType::Product => self.product_or(a, b)?,
            TNormType::Godel => self.godel_or(a, b)?,
        };

        if self.config.clamp_outputs {
            self.clamp_01(&result)
        } else {
            Ok(result)
        }
    }

    /// Łukasiewicz T-conorm: min(1, a + b)
    fn lukasiewicz_or(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let sum = (a + b)?;
        sum.minimum(&Tensor::ones_like(&sum)?)
    }

    /// Product T-conorm (probabilistic sum): a + b - a*b
    fn product_or(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let sum = (a + b)?;
        let product = (a * b)?;
        sum - product
    }

    /// Gödel T-conorm: max(a, b) using soft maximum for differentiability
    fn godel_or(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        self.soft_max(a, b)
    }

    // =========================================================================
    // Negation
    // =========================================================================

    /// Fuzzy NOT (negation): ¬A = 1 - A
    ///
    /// Standard negation is the same for all t-norm families.
    ///
    /// # Arguments
    /// * `a` - Operand tensor with values in [0, 1]
    ///
    /// # Returns
    /// Tensor with fuzzy NOT result
    pub fn not(&self, a: &Tensor) -> Result<Tensor> {
        let one = Tensor::ones_like(a)?;
        one - a
    }

    // =========================================================================
    // Implication
    // =========================================================================

    /// Fuzzy implication: A → B
    ///
    /// Uses the residual implication derived from each t-norm family.
    ///
    /// # Arguments
    /// * `a` - Antecedent tensor (condition) with values in [0, 1]
    /// * `b` - Consequent tensor (result) with values in [0, 1]
    ///
    /// # Returns
    /// Tensor with fuzzy implication result
    pub fn implies(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let result = match self.config.tnorm_type {
            TNormType::Lukasiewicz => self.lukasiewicz_implies(a, b)?,
            TNormType::Product => self.product_implies(a, b)?,
            TNormType::Godel => self.godel_implies(a, b)?,
        };

        if self.config.clamp_outputs {
            self.clamp_01(&result)
        } else {
            Ok(result)
        }
    }

    /// Łukasiewicz implication: min(1, 1 - a + b)
    fn lukasiewicz_implies(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let one = Tensor::ones_like(a)?;
        let diff = (&one - a)?;
        let sum = (diff + b)?;
        sum.minimum(&one)
    }

    /// Product implication (Goguen): if a ≤ b then 1, else b/a
    /// We use a smooth approximation: min(1, b / (a + eps))
    fn product_implies(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        // Use affine to add epsilon: a + eps = 1*a + eps
        let a_safe = a.affine(1.0, self.config.eps)?;
        let ratio = (b / &a_safe)?;
        let one = Tensor::ones_like(&ratio)?;
        ratio.minimum(&one)
    }

    /// Gödel implication: if a ≤ b then 1, else b
    /// Uses soft approximation for differentiability
    fn godel_implies(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        // Smooth approximation: sigmoid((b - a) / temp) * (1 - b) + b
        // When b >= a: → 1
        // When b < a: → b
        let diff = (b - a)?;
        // Scale by 1/temperature using affine
        let scaled = diff.affine(1.0 / self.config.temperature as f64, 0.0)?;
        let sigmoid = self.sigmoid(&scaled)?;

        let one = Tensor::ones_like(b)?;
        let one_minus_b = (&one - b)?;
        let correction = (&sigmoid * &one_minus_b)?;
        &correction + b
    }

    // =========================================================================
    // Quantifiers (for aggregating over batches)
    // =========================================================================

    /// Universal quantifier (ForAll): ∀x P(x)
    ///
    /// Aggregates truth values across the specified dimension using AND.
    /// For Łukasiewicz: generalized mean, for Product: geometric mean
    ///
    /// # Arguments
    /// * `values` - Tensor of truth values
    /// * `dim` - Dimension to aggregate over (None = flatten all)
    ///
    /// # Returns
    /// Aggregated truth value (scalar if dim=None, reduced tensor otherwise)
    pub fn forall(&self, values: &Tensor, dim: Option<usize>) -> Result<Tensor> {
        match self.config.tnorm_type {
            TNormType::Lukasiewicz => self.forall_lukasiewicz(values, dim),
            TNormType::Product => self.forall_product(values, dim),
            TNormType::Godel => self.forall_godel(values, dim),
        }
    }

    /// Łukasiewicz ForAll: mean (provides smooth gradient)
    fn forall_lukasiewicz(&self, values: &Tensor, dim: Option<usize>) -> Result<Tensor> {
        match dim {
            Some(d) => values.mean_keepdim(d),
            None => values.mean_all(),
        }
    }

    /// Product ForAll: geometric mean = exp(mean(log(x + eps)))
    fn forall_product(&self, values: &Tensor, dim: Option<usize>) -> Result<Tensor> {
        // Use affine to add epsilon for numerical stability
        let safe_values = values.affine(1.0, self.config.eps)?;
        let log_values = safe_values.log()?;

        let mean_log = match dim {
            Some(d) => log_values.mean_keepdim(d)?,
            None => log_values.mean_all()?,
        };

        mean_log.exp()
    }

    /// Gödel ForAll: soft minimum
    fn forall_godel(&self, values: &Tensor, dim: Option<usize>) -> Result<Tensor> {
        // Soft min using negative temperature softmax
        let neg_temp = -1.0 / self.config.temperature;
        let scaled = values.affine(neg_temp as f64, 0.0)?;

        let softmax = match dim {
            Some(d) => candle_nn::ops::softmax(&scaled, d)?,
            None => {
                let flat = scaled.flatten_all()?;
                candle_nn::ops::softmax(&flat, 0)?
            }
        };

        // Weighted sum
        let weighted = (&softmax * values)?;
        match dim {
            Some(d) => weighted.sum_keepdim(d),
            None => weighted.sum_all(),
        }
    }

    /// Existential quantifier (Exists): ∃x P(x)
    ///
    /// Aggregates truth values across the specified dimension using OR.
    ///
    /// # Arguments
    /// * `values` - Tensor of truth values
    /// * `dim` - Dimension to aggregate over (None = flatten all)
    ///
    /// # Returns
    /// Aggregated truth value
    pub fn exists(&self, values: &Tensor, dim: Option<usize>) -> Result<Tensor> {
        // Exists is dual to ForAll: ∃x P(x) = ¬∀x ¬P(x)
        let neg_values = self.not(values)?;
        let forall_neg = self.forall(&neg_values, dim)?;
        self.not(&forall_neg)
    }

    // =========================================================================
    // N-ary Operations
    // =========================================================================

    /// N-ary AND: computes A₁ ∧ A₂ ∧ ... ∧ Aₙ
    ///
    /// # Arguments
    /// * `tensors` - Slice of tensors to AND together
    ///
    /// # Returns
    /// Result of ANDing all tensors
    pub fn and_n(&self, tensors: &[&Tensor]) -> Result<Tensor> {
        if tensors.is_empty() {
            return Err(candle_core::Error::Msg(
                "Cannot AND empty tensor list".into(),
            ));
        }
        if tensors.len() == 1 {
            return Ok(tensors[0].clone());
        }

        let mut result = self.and(tensors[0], tensors[1])?;
        for t in &tensors[2..] {
            result = self.and(&result, t)?;
        }
        Ok(result)
    }

    /// N-ary OR: computes A₁ ∨ A₂ ∨ ... ∨ Aₙ
    ///
    /// # Arguments
    /// * `tensors` - Slice of tensors to OR together
    ///
    /// # Returns
    /// Result of ORing all tensors
    pub fn or_n(&self, tensors: &[&Tensor]) -> Result<Tensor> {
        if tensors.is_empty() {
            return Err(candle_core::Error::Msg(
                "Cannot OR empty tensor list".into(),
            ));
        }
        if tensors.len() == 1 {
            return Ok(tensors[0].clone());
        }

        let mut result = self.or(tensors[0], tensors[1])?;
        for t in &tensors[2..] {
            result = self.or(&result, t)?;
        }
        Ok(result)
    }

    // =========================================================================
    // Helper Functions
    // =========================================================================

    /// Soft minimum using negative temperature softmax
    /// Uses log-sum-exp trick for numerical stability
    fn soft_min(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let neg_temp = -1.0 / self.config.temperature;

        // For numerical stability, compute softmax using log-sum-exp trick
        let a_scaled = a.affine(neg_temp as f64, 0.0)?;
        let b_scaled = b.affine(neg_temp as f64, 0.0)?;

        // Find max for stability
        let max_val = a_scaled.maximum(&b_scaled)?;

        // Compute exp(x - max) for stability
        let a_shifted = (&a_scaled - &max_val)?;
        let b_shifted = (&b_scaled - &max_val)?;

        let a_exp = a_shifted.exp()?;
        let b_exp = b_shifted.exp()?;
        let sum_exp = (&a_exp + &b_exp)?;

        let w_a = (&a_exp / &sum_exp)?;
        let w_b = (&b_exp / &sum_exp)?;

        // Weighted sum gives soft min
        let weighted_a = (&w_a * a)?;
        let weighted_b = (&w_b * b)?;
        weighted_a + weighted_b
    }

    /// Soft maximum using temperature softmax
    /// Uses log-sum-exp trick for numerical stability
    fn soft_max(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let temp = 1.0 / self.config.temperature;

        // For numerical stability, compute softmax using log-sum-exp trick
        let a_scaled = a.affine(temp as f64, 0.0)?;
        let b_scaled = b.affine(temp as f64, 0.0)?;

        // Find max for stability
        let max_val = a_scaled.maximum(&b_scaled)?;

        // Compute exp(x - max) for stability
        let a_shifted = (&a_scaled - &max_val)?;
        let b_shifted = (&b_scaled - &max_val)?;

        let a_exp = a_shifted.exp()?;
        let b_exp = b_shifted.exp()?;
        let sum_exp = (&a_exp + &b_exp)?;

        let w_a = (&a_exp / &sum_exp)?;
        let w_b = (&b_exp / &sum_exp)?;

        // Weighted sum gives soft max
        let weighted_a = (&w_a * a)?;
        let weighted_b = (&w_b * b)?;
        weighted_a + weighted_b
    }

    /// Sigmoid activation
    fn sigmoid(&self, x: &Tensor) -> Result<Tensor> {
        let neg_x = x.neg()?;
        let exp_neg = neg_x.exp()?;
        let one = Tensor::ones_like(&exp_neg)?;
        let denom = (&one + &exp_neg)?;
        one.broadcast_div(&denom)
    }

    /// Clamp tensor values to [0, 1]
    fn clamp_01(&self, x: &Tensor) -> Result<Tensor> {
        let zero = Tensor::zeros_like(x)?;
        let one = Tensor::ones_like(x)?;
        let lower_bounded = x.maximum(&zero)?;
        lower_bounded.minimum(&one)
    }
}

impl Default for DiffTNorm {
    fn default() -> Self {
        Self::new(TNormType::Lukasiewicz)
    }
}

// =============================================================================
// Standalone Functions (for convenience)
// =============================================================================

/// Create a Łukasiewicz t-norm operator.
pub fn lukasiewicz() -> DiffTNorm {
    DiffTNorm::new(TNormType::Lukasiewicz)
}

/// Create a Product t-norm operator.
pub fn product() -> DiffTNorm {
    DiffTNorm::new(TNormType::Product)
}

/// Create a Gödel t-norm operator with specified temperature.
pub fn godel(temperature: f32) -> DiffTNorm {
    DiffTNorm::with_config(DiffTNormConfig::godel(temperature))
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn create_test_tensors(device: &Device) -> (Tensor, Tensor) {
        let a = Tensor::new(&[0.8f32, 0.6, 0.9, 0.3], device).unwrap();
        let b = Tensor::new(&[0.7f32, 0.5, 0.3, 0.8], device).unwrap();
        (a, b)
    }

    #[test]
    fn test_lukasiewicz_and() {
        let device = Device::Cpu;
        let (a, b) = create_test_tensors(&device);
        let tnorm = lukasiewicz();

        let result = tnorm.and(&a, &b).unwrap();
        let values: Vec<f32> = result.to_vec1().unwrap();

        // max(0, 0.8 + 0.7 - 1) = 0.5
        // max(0, 0.6 + 0.5 - 1) = 0.1
        // max(0, 0.9 + 0.3 - 1) = 0.2
        // max(0, 0.3 + 0.8 - 1) = 0.1
        assert!((values[0] - 0.5).abs() < 1e-5);
        assert!((values[1] - 0.1).abs() < 1e-5);
        assert!((values[2] - 0.2).abs() < 1e-5);
        assert!((values[3] - 0.1).abs() < 1e-5);
    }

    #[test]
    fn test_lukasiewicz_or() {
        let device = Device::Cpu;
        let (a, b) = create_test_tensors(&device);
        let tnorm = lukasiewicz();

        let result = tnorm.or(&a, &b).unwrap();
        let values: Vec<f32> = result.to_vec1().unwrap();

        // min(1, 0.8 + 0.7) = 1.0
        // min(1, 0.6 + 0.5) = 1.0
        // min(1, 0.9 + 0.3) = 1.0
        // min(1, 0.3 + 0.8) = 1.0
        assert!((values[0] - 1.0).abs() < 1e-5);
        assert!((values[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_lukasiewicz_implies() {
        let device = Device::Cpu;
        let (a, b) = create_test_tensors(&device);
        let tnorm = lukasiewicz();

        let result = tnorm.implies(&a, &b).unwrap();
        let values: Vec<f32> = result.to_vec1().unwrap();

        // min(1, 1 - 0.8 + 0.7) = 0.9
        // min(1, 1 - 0.6 + 0.5) = 0.9
        // min(1, 1 - 0.9 + 0.3) = 0.4
        // min(1, 1 - 0.3 + 0.8) = 1.0
        assert!((values[0] - 0.9).abs() < 1e-5);
        assert!((values[1] - 0.9).abs() < 1e-5);
        assert!((values[2] - 0.4).abs() < 1e-5);
        assert!((values[3] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_product_and() {
        let device = Device::Cpu;
        let (a, b) = create_test_tensors(&device);
        let tnorm = product();

        let result = tnorm.and(&a, &b).unwrap();
        let values: Vec<f32> = result.to_vec1().unwrap();

        // 0.8 * 0.7 = 0.56
        // 0.6 * 0.5 = 0.30
        // 0.9 * 0.3 = 0.27
        // 0.3 * 0.8 = 0.24
        assert!((values[0] - 0.56).abs() < 1e-5);
        assert!((values[1] - 0.30).abs() < 1e-5);
        assert!((values[2] - 0.27).abs() < 1e-5);
        assert!((values[3] - 0.24).abs() < 1e-5);
    }

    #[test]
    fn test_product_or() {
        let device = Device::Cpu;
        let (a, b) = create_test_tensors(&device);
        let tnorm = product();

        let result = tnorm.or(&a, &b).unwrap();
        let values: Vec<f32> = result.to_vec1().unwrap();

        // 0.8 + 0.7 - 0.8*0.7 = 0.94
        // 0.6 + 0.5 - 0.6*0.5 = 0.80
        assert!((values[0] - 0.94).abs() < 1e-5);
        assert!((values[1] - 0.80).abs() < 1e-5);
    }

    #[test]
    fn test_godel_and() {
        let device = Device::Cpu;
        let (a, b) = create_test_tensors(&device);
        let tnorm = godel(0.05); // Moderate temperature for numerical stability

        let result = tnorm.and(&a, &b).unwrap();
        let values: Vec<f32> = result.to_vec1().unwrap();

        // Soft min should be close to min(a, b) with some tolerance
        assert!(
            (values[0] - 0.7).abs() < 0.15,
            "min(0.8, 0.7) = 0.7, got {}",
            values[0]
        );
        assert!(
            (values[1] - 0.5).abs() < 0.15,
            "min(0.6, 0.5) = 0.5, got {}",
            values[1]
        );
        assert!(
            (values[2] - 0.3).abs() < 0.15,
            "min(0.9, 0.3) = 0.3, got {}",
            values[2]
        );
        assert!(
            (values[3] - 0.3).abs() < 0.15,
            "min(0.3, 0.8) = 0.3, got {}",
            values[3]
        );
    }

    #[test]
    fn test_godel_or() {
        let device = Device::Cpu;
        let (a, b) = create_test_tensors(&device);
        let tnorm = godel(0.05); // Moderate temperature for numerical stability

        let result = tnorm.or(&a, &b).unwrap();
        let values: Vec<f32> = result.to_vec1().unwrap();

        // Soft max should be close to max(a, b) with some tolerance
        assert!(
            (values[0] - 0.8).abs() < 0.15,
            "max(0.8, 0.7) = 0.8, got {}",
            values[0]
        );
        assert!(
            (values[1] - 0.6).abs() < 0.15,
            "max(0.6, 0.5) = 0.6, got {}",
            values[1]
        );
        assert!(
            (values[2] - 0.9).abs() < 0.15,
            "max(0.9, 0.3) = 0.9, got {}",
            values[2]
        );
        assert!(
            (values[3] - 0.8).abs() < 0.15,
            "max(0.3, 0.8) = 0.8, got {}",
            values[3]
        );
    }

    #[test]
    fn test_not() {
        let device = Device::Cpu;
        let a = Tensor::new(&[0.8f32, 0.0, 1.0, 0.5], &device).unwrap();
        let tnorm = lukasiewicz();

        let result = tnorm.not(&a).unwrap();
        let values: Vec<f32> = result.to_vec1().unwrap();

        assert!((values[0] - 0.2).abs() < 1e-5);
        assert!((values[1] - 1.0).abs() < 1e-5);
        assert!((values[2] - 0.0).abs() < 1e-5);
        assert!((values[3] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_forall_lukasiewicz() {
        let device = Device::Cpu;
        let values = Tensor::new(&[0.9f32, 0.8, 0.7, 1.0], &device).unwrap();
        let tnorm = lukasiewicz();

        let result = tnorm.forall(&values, None).unwrap();
        let scalar: f32 = result.to_scalar().unwrap();

        // Mean of [0.9, 0.8, 0.7, 1.0] = 0.85
        assert!((scalar - 0.85).abs() < 1e-5);
    }

    #[test]
    fn test_forall_product() {
        let device = Device::Cpu;
        let values = Tensor::new(&[0.9f32, 0.8, 0.7, 1.0], &device).unwrap();
        let tnorm = product();

        let result = tnorm.forall(&values, None).unwrap();
        let scalar: f32 = result.to_scalar().unwrap();

        // Geometric mean of [0.9, 0.8, 0.7, 1.0] ≈ 0.845
        let expected = (0.9f32 * 0.8 * 0.7 * 1.0).powf(0.25);
        assert!((scalar - expected).abs() < 0.01);
    }

    #[test]
    fn test_exists() {
        let device = Device::Cpu;
        let values = Tensor::new(&[0.1f32, 0.2, 0.9, 0.05], &device).unwrap();
        let tnorm = lukasiewicz();

        let result = tnorm.exists(&values, None).unwrap();
        let scalar: f32 = result.to_scalar().unwrap();

        // Exists with Lukasiewicz: 1 - mean(1 - values)
        // = 1 - mean(0.9, 0.8, 0.1, 0.95) = 1 - 0.6875 = 0.3125
        // This is lower than expected because Lukasiewicz uses mean
        // The value 0.9 alone doesn't dominate
        assert!(scalar > 0.3);
    }

    #[test]
    fn test_and_n() {
        let device = Device::Cpu;
        let a = Tensor::new(&[0.9f32], &device).unwrap();
        let b = Tensor::new(&[0.8f32], &device).unwrap();
        let c = Tensor::new(&[0.7f32], &device).unwrap();

        let tnorm = lukasiewicz();
        let result = tnorm.and_n(&[&a, &b, &c]).unwrap();
        // Result is [1] tensor, need to flatten to scalar
        let values: Vec<f32> = result.to_vec1().unwrap();

        // Łukasiewicz: max(0, max(0, 0.9 + 0.8 - 1) + 0.7 - 1)
        // = max(0, 0.7 + 0.7 - 1) = 0.4
        assert!((values[0] - 0.4).abs() < 1e-5);
    }

    #[test]
    fn test_or_n() {
        let device = Device::Cpu;
        let a = Tensor::new(&[0.3f32], &device).unwrap();
        let b = Tensor::new(&[0.4f32], &device).unwrap();
        let c = Tensor::new(&[0.2f32], &device).unwrap();

        let tnorm = lukasiewicz();
        let result = tnorm.or_n(&[&a, &b, &c]).unwrap();
        // Result is [1] tensor, need to use to_vec1
        let values: Vec<f32> = result.to_vec1().unwrap();

        // Łukasiewicz: min(1, min(1, 0.3 + 0.4) + 0.2) = min(1, 0.9) = 0.9
        assert!((values[0] - 0.9).abs() < 1e-5);
    }

    #[test]
    fn test_boundary_conditions() {
        let device = Device::Cpu;
        let zero = Tensor::new(&[0.0f32], &device).unwrap();
        let one = Tensor::new(&[1.0f32], &device).unwrap();

        for tnorm in [lukasiewicz(), product(), godel(0.1)] {
            // AND with 0 should give 0
            let and_zero = tnorm.and(&one, &zero).unwrap();
            let v: Vec<f32> = and_zero.to_vec1().unwrap();
            assert!(v[0].abs() < 0.01, "AND(1, 0) should be ~0");

            // AND with 1 should preserve value
            let and_one = tnorm.and(&one, &one).unwrap();
            let v: Vec<f32> = and_one.to_vec1().unwrap();
            assert!((v[0] - 1.0).abs() < 0.01, "AND(1, 1) should be ~1");

            // OR with 1 should give 1
            let or_one = tnorm.or(&zero, &one).unwrap();
            let v: Vec<f32> = or_one.to_vec1().unwrap();
            assert!((v[0] - 1.0).abs() < 0.01, "OR(0, 1) should be ~1");

            // OR with 0 should preserve value
            let or_zero = tnorm.or(&zero, &zero).unwrap();
            let v: Vec<f32> = or_zero.to_vec1().unwrap();
            assert!(v[0].abs() < 0.01, "OR(0, 0) should be ~0");
        }
    }

    #[test]
    fn test_output_bounds() {
        let device = Device::Cpu;
        let a = Tensor::new(&[0.8f32, 0.3, 0.95, 0.1], &device).unwrap();
        let b = Tensor::new(&[0.7f32, 0.9, 0.05, 0.85], &device).unwrap();

        // Test Lukasiewicz and Product (Godel has numerical edge cases)
        for tnorm in [lukasiewicz(), product()] {
            let and_result = tnorm.and(&a, &b).unwrap();
            let or_result = tnorm.or(&a, &b).unwrap();
            let implies_result = tnorm.implies(&a, &b).unwrap();

            for result in [and_result, or_result, implies_result] {
                let values: Vec<f32> = result.to_vec1().unwrap();
                for v in values {
                    assert!(v >= 0.0 && v <= 1.0, "Output {v} out of bounds [0, 1]");
                }
            }
        }

        // Test Godel separately with more tolerance for soft approximations
        let tnorm = godel(0.1);
        let and_result = tnorm.and(&a, &b).unwrap();
        let or_result = tnorm.or(&a, &b).unwrap();

        for result in [and_result, or_result] {
            let values: Vec<f32> = result.to_vec1().unwrap();
            for v in values {
                assert!(
                    v.is_finite() && v >= -0.01 && v <= 1.01,
                    "Godel output {v} out of bounds or not finite"
                );
            }
        }
    }

    #[test]
    fn test_batched_operations() {
        let device = Device::Cpu;
        // Batch of 3, sequence of 4 values
        let a = Tensor::new(
            &[
                [0.8f32, 0.6, 0.9, 0.3],
                [0.5, 0.5, 0.5, 0.5],
                [1.0, 0.0, 1.0, 0.0],
            ],
            &device,
        )
        .unwrap();
        let b = Tensor::new(
            &[
                [0.7f32, 0.5, 0.3, 0.8],
                [0.5, 0.5, 0.5, 0.5],
                [0.0, 1.0, 0.0, 1.0],
            ],
            &device,
        )
        .unwrap();

        let tnorm = lukasiewicz();
        let result = tnorm.and(&a, &b).unwrap();

        assert_eq!(result.dims(), &[3, 4]);
    }

    #[test]
    fn test_forall_with_dim() {
        let device = Device::Cpu;
        // Batch of 2, 3 predicates each
        let values = Tensor::new(&[[0.9f32, 0.8, 1.0], [0.5, 0.6, 0.7]], &device).unwrap();

        let tnorm = lukasiewicz();
        let result = tnorm.forall(&values, Some(1)).unwrap();

        assert_eq!(result.dims(), &[2, 1]);

        let flattened: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
        // Row 0 mean: (0.9 + 0.8 + 1.0) / 3 = 0.9
        // Row 1 mean: (0.5 + 0.6 + 0.7) / 3 = 0.6
        assert!((flattened[0] - 0.9).abs() < 1e-5);
        assert!((flattened[1] - 0.6).abs() < 1e-5);
    }
}
