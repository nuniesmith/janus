//! Logic Tensor Networks (LTN) - Differentiable Neuro-Symbolic Integration.
//!
//! This module provides the core LTN framework for integrating neural networks with
//! logical reasoning. It enables:
//!
//! 1. **Logical Constraints as Losses**: Convert logical rules into differentiable loss functions
//! 2. **Grounding**: Map neural network outputs to logical variables
//! 3. **Rule Satisfaction**: Measure and optimize how well neural outputs satisfy logical rules
//!
//! # Architecture
//!
//! ```text
//! Neural Network → Grounding → Predicates → Logical Rules → Satisfaction Loss
//!       ↓              ↓            ↓              ↓               ↓
//!   Embeddings    Variables   Truth Values   Formula Eval    Backprop
//! ```
//!
//! # Example
//!
//! ```ignore
//! use logic::ltn::{DiffLTN, LogicalRule, RuleBuilder};
//! use logic::diff_tnorm::TNormType;
//! use logic::predicates::ThresholdPredicate;
//! use candle_core::{Device, Tensor};
//!
//! let device = Device::Cpu;
//! let mut ltn = DiffLTN::new(TNormType::Lukasiewicz);
//!
//! // Define rules
//! // Rule 1: If confidence is high, action should be taken
//! // Rule 2: Position size should be within bounds
//! ltn.add_rule(
//!     RuleBuilder::new("high_conf_implies_action")
//!         .implies("confidence_high", "action_taken")
//!         .build()
//! );
//!
//! // Compute satisfaction loss
//! let grounding = Grounding::new();
//! grounding.set("confidence_high", confidence_tensor);
//! grounding.set("action_taken", action_tensor);
//!
//! let loss = ltn.satisfaction_loss(&grounding)?;
//! // Backpropagate through the loss to train the neural network
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::{Device, Result, Tensor};
use serde::{Deserialize, Serialize};

use crate::diff_tnorm::{DiffTNorm, DiffTNormConfig, TNormType};
use crate::predicates::LearnablePredicate;

// =============================================================================
// Grounding: Maps variables to tensors
// =============================================================================

/// A grounding maps variable names to tensor values.
///
/// This connects neural network outputs to logical variables used in rules.
#[derive(Debug, Clone, Default)]
pub struct Grounding {
    variables: HashMap<String, Tensor>,
}

impl Grounding {
    /// Create a new empty grounding.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a variable's value.
    pub fn set(&mut self, name: &str, value: Tensor) {
        self.variables.insert(name.to_string(), value);
    }

    /// Get a variable's value.
    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.variables.get(name)
    }

    /// Check if a variable exists.
    pub fn contains(&self, name: &str) -> bool {
        self.variables.contains_key(name)
    }

    /// Get all variable names.
    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.variables.keys()
    }

    /// Get the number of variables.
    pub fn len(&self) -> usize {
        self.variables.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.variables.is_empty()
    }

    /// Create grounding from iterator.
    pub fn from_iter<I, S>(iter: I) -> Self
    where
        I: IntoIterator<Item = (S, Tensor)>,
        S: Into<String>,
    {
        let variables = iter.into_iter().map(|(k, v)| (k.into(), v)).collect();
        Self { variables }
    }
}

// =============================================================================
// Logical Formulas
// =============================================================================

/// A logical formula that can be evaluated given a grounding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Formula {
    /// A variable reference
    Var(String),

    /// Constant truth value
    Const(f32),

    /// Negation: ¬A
    Not(Box<Formula>),

    /// Conjunction: A ∧ B
    And(Box<Formula>, Box<Formula>),

    /// Disjunction: A ∨ B
    Or(Box<Formula>, Box<Formula>),

    /// Implication: A → B
    Implies(Box<Formula>, Box<Formula>),

    /// Biconditional: A ↔ B (A → B) ∧ (B → A)
    Iff(Box<Formula>, Box<Formula>),

    /// N-ary conjunction: A₁ ∧ A₂ ∧ ... ∧ Aₙ
    AndN(Vec<Formula>),

    /// N-ary disjunction: A₁ ∨ A₂ ∨ ... ∨ Aₙ
    OrN(Vec<Formula>),

    /// Universal quantifier over a dimension: ∀x P(x)
    ForAll(Box<Formula>, Option<usize>),

    /// Existential quantifier over a dimension: ∃x P(x)
    Exists(Box<Formula>, Option<usize>),
}

impl Formula {
    /// Create a variable reference.
    pub fn var(name: &str) -> Self {
        Formula::Var(name.to_string())
    }

    /// Create a constant truth value.
    pub fn constant(value: f32) -> Self {
        Formula::Const(value.clamp(0.0, 1.0))
    }

    /// Create TRUE constant.
    pub fn true_const() -> Self {
        Formula::Const(1.0)
    }

    /// Create FALSE constant.
    pub fn false_const() -> Self {
        Formula::Const(0.0)
    }

    /// Negate this formula: ¬self
    pub fn not(self) -> Self {
        Formula::Not(Box::new(self))
    }

    /// Conjoin with another formula: self ∧ other
    pub fn and(self, other: Formula) -> Self {
        Formula::And(Box::new(self), Box::new(other))
    }

    /// Disjoin with another formula: self ∨ other
    pub fn or(self, other: Formula) -> Self {
        Formula::Or(Box::new(self), Box::new(other))
    }

    /// Implication: self → other
    pub fn implies(self, other: Formula) -> Self {
        Formula::Implies(Box::new(self), Box::new(other))
    }

    /// Biconditional: self ↔ other
    pub fn iff(self, other: Formula) -> Self {
        Formula::Iff(Box::new(self), Box::new(other))
    }

    /// Universal quantifier over this formula.
    pub fn forall(self, dim: Option<usize>) -> Self {
        Formula::ForAll(Box::new(self), dim)
    }

    /// Existential quantifier over this formula.
    pub fn exists(self, dim: Option<usize>) -> Self {
        Formula::Exists(Box::new(self), dim)
    }

    /// Evaluate this formula given a grounding and t-norm.
    pub fn evaluate(&self, grounding: &Grounding, tnorm: &DiffTNorm) -> Result<Tensor> {
        match self {
            Formula::Var(name) => grounding.get(name).cloned().ok_or_else(|| {
                candle_core::Error::Msg(format!("Variable '{name}' not found in grounding"))
            }),

            Formula::Const(value) => {
                // Get device from any variable in grounding, or use CPU
                let device = grounding
                    .variables
                    .values()
                    .next()
                    .map(|t| t.device().clone())
                    .unwrap_or(Device::Cpu);
                Tensor::new(*value, &device)
            }

            Formula::Not(inner) => {
                let inner_val = inner.evaluate(grounding, tnorm)?;
                tnorm.not(&inner_val)
            }

            Formula::And(a, b) => {
                let a_val = a.evaluate(grounding, tnorm)?;
                let b_val = b.evaluate(grounding, tnorm)?;
                tnorm.and(&a_val, &b_val)
            }

            Formula::Or(a, b) => {
                let a_val = a.evaluate(grounding, tnorm)?;
                let b_val = b.evaluate(grounding, tnorm)?;
                tnorm.or(&a_val, &b_val)
            }

            Formula::Implies(a, b) => {
                let a_val = a.evaluate(grounding, tnorm)?;
                let b_val = b.evaluate(grounding, tnorm)?;
                tnorm.implies(&a_val, &b_val)
            }

            Formula::Iff(a, b) => {
                // A ↔ B = (A → B) ∧ (B → A)
                let a_val = a.evaluate(grounding, tnorm)?;
                let b_val = b.evaluate(grounding, tnorm)?;
                let a_implies_b = tnorm.implies(&a_val, &b_val)?;
                let b_implies_a = tnorm.implies(&b_val, &a_val)?;
                tnorm.and(&a_implies_b, &b_implies_a)
            }

            Formula::AndN(formulas) => {
                if formulas.is_empty() {
                    return Formula::true_const().evaluate(grounding, tnorm);
                }
                let tensors: Vec<Tensor> = formulas
                    .iter()
                    .map(|f| f.evaluate(grounding, tnorm))
                    .collect::<Result<Vec<_>>>()?;
                let refs: Vec<&Tensor> = tensors.iter().collect();
                tnorm.and_n(&refs)
            }

            Formula::OrN(formulas) => {
                if formulas.is_empty() {
                    return Formula::false_const().evaluate(grounding, tnorm);
                }
                let tensors: Vec<Tensor> = formulas
                    .iter()
                    .map(|f| f.evaluate(grounding, tnorm))
                    .collect::<Result<Vec<_>>>()?;
                let refs: Vec<&Tensor> = tensors.iter().collect();
                tnorm.or_n(&refs)
            }

            Formula::ForAll(inner, dim) => {
                let inner_val = inner.evaluate(grounding, tnorm)?;
                tnorm.forall(&inner_val, *dim)
            }

            Formula::Exists(inner, dim) => {
                let inner_val = inner.evaluate(grounding, tnorm)?;
                tnorm.exists(&inner_val, *dim)
            }
        }
    }

    /// Get all variable names referenced in this formula.
    pub fn variables(&self) -> Vec<String> {
        let mut vars = Vec::new();
        self.collect_variables(&mut vars);
        vars.sort();
        vars.dedup();
        vars
    }

    fn collect_variables(&self, vars: &mut Vec<String>) {
        match self {
            Formula::Var(name) => vars.push(name.clone()),
            Formula::Const(_) => {}
            Formula::Not(inner) => inner.collect_variables(vars),
            Formula::And(a, b)
            | Formula::Or(a, b)
            | Formula::Implies(a, b)
            | Formula::Iff(a, b) => {
                a.collect_variables(vars);
                b.collect_variables(vars);
            }
            Formula::AndN(formulas) | Formula::OrN(formulas) => {
                for f in formulas {
                    f.collect_variables(vars);
                }
            }
            Formula::ForAll(inner, _) | Formula::Exists(inner, _) => {
                inner.collect_variables(vars);
            }
        }
    }
}

// =============================================================================
// Logical Rule
// =============================================================================

/// A named logical rule with an optional weight.
#[derive(Debug, Clone)]
pub struct LogicalRule {
    /// Name of the rule for identification
    pub name: String,

    /// The logical formula
    pub formula: Formula,

    /// Weight of this rule in the total loss (default: 1.0)
    pub weight: f32,

    /// Whether this rule is active
    pub active: bool,
}

impl LogicalRule {
    /// Create a new rule.
    pub fn new(name: &str, formula: Formula) -> Self {
        Self {
            name: name.to_string(),
            formula,
            weight: 1.0,
            active: true,
        }
    }

    /// Set the weight of this rule.
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }

    /// Set whether this rule is active.
    pub fn with_active(mut self, active: bool) -> Self {
        self.active = active;
        self
    }

    /// Evaluate this rule's satisfaction.
    pub fn evaluate(&self, grounding: &Grounding, tnorm: &DiffTNorm) -> Result<Tensor> {
        self.formula.evaluate(grounding, tnorm)
    }
}

// =============================================================================
// Rule Builder
// =============================================================================

/// Builder for creating logical rules with a fluent API.
pub struct RuleBuilder {
    name: String,
    formula: Option<Formula>,
    weight: f32,
}

impl RuleBuilder {
    /// Create a new rule builder.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            formula: None,
            weight: 1.0,
        }
    }

    /// Set the formula directly.
    pub fn formula(mut self, formula: Formula) -> Self {
        self.formula = Some(formula);
        self
    }

    /// Create an implication rule: antecedent → consequent
    pub fn implies(mut self, antecedent: &str, consequent: &str) -> Self {
        self.formula = Some(Formula::var(antecedent).implies(Formula::var(consequent)));
        self
    }

    /// Create a conjunction rule: a ∧ b
    pub fn and(mut self, a: &str, b: &str) -> Self {
        self.formula = Some(Formula::var(a).and(Formula::var(b)));
        self
    }

    /// Create a disjunction rule: a ∨ b
    pub fn or(mut self, a: &str, b: &str) -> Self {
        self.formula = Some(Formula::var(a).or(Formula::var(b)));
        self
    }

    /// Create a biconditional rule: a ↔ b
    pub fn iff(mut self, a: &str, b: &str) -> Self {
        self.formula = Some(Formula::var(a).iff(Formula::var(b)));
        self
    }

    /// Create a negation rule: ¬a
    pub fn not(mut self, a: &str) -> Self {
        self.formula = Some(Formula::var(a).not());
        self
    }

    /// Create a rule that requires a variable to be true.
    pub fn requires(mut self, var: &str) -> Self {
        self.formula = Some(Formula::var(var));
        self
    }

    /// Create a universal quantification: ∀x P(x)
    pub fn forall(mut self, var: &str) -> Self {
        self.formula = Some(Formula::var(var).forall(None));
        self
    }

    /// Create an existential quantification: ∃x P(x)
    pub fn exists(mut self, var: &str) -> Self {
        self.formula = Some(Formula::var(var).exists(None));
        self
    }

    /// Set the weight.
    pub fn weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }

    /// Build the rule.
    pub fn build(self) -> LogicalRule {
        LogicalRule {
            name: self.name,
            formula: self.formula.unwrap_or(Formula::true_const()),
            weight: self.weight,
            active: true,
        }
    }
}

// =============================================================================
// LTN Configuration
// =============================================================================

/// Configuration for the differentiable LTN.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffLTNConfig {
    /// T-norm configuration for fuzzy logic operations
    pub tnorm_config: DiffTNormConfig,

    /// Aggregation method for combining rule satisfactions
    pub aggregation: SatisfactionAggregation,

    /// Whether to normalize rule weights
    pub normalize_weights: bool,
}

impl Default for DiffLTNConfig {
    fn default() -> Self {
        Self {
            tnorm_config: DiffTNormConfig::default(),
            aggregation: SatisfactionAggregation::Mean,
            normalize_weights: true,
        }
    }
}

/// Method for aggregating satisfaction across multiple rules.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum SatisfactionAggregation {
    /// Weighted mean of satisfactions
    #[default]
    Mean,

    /// Product of satisfactions (stricter)
    Product,

    /// Minimum satisfaction (strictest)
    Min,

    /// Weighted sum (not normalized)
    Sum,
}

// =============================================================================
// Differentiable LTN
// =============================================================================

/// Differentiable Logic Tensor Network.
///
/// Combines neural network outputs with logical rules to produce differentiable
/// satisfaction losses that can be backpropagated for training.
pub struct DiffLTN {
    /// Configuration
    config: DiffLTNConfig,

    /// T-norm operator for fuzzy logic
    tnorm: DiffTNorm,

    /// Registered rules
    rules: Vec<LogicalRule>,

    /// Learnable predicates (optional)
    predicates: HashMap<String, Arc<LearnablePredicate>>,
}

impl DiffLTN {
    /// Create a new DiffLTN with the specified t-norm type.
    pub fn new(tnorm_type: TNormType) -> Self {
        let config = DiffLTNConfig {
            tnorm_config: DiffTNormConfig {
                tnorm_type,
                ..Default::default()
            },
            ..Default::default()
        };
        Self::with_config(config)
    }

    /// Create a new DiffLTN with full configuration.
    pub fn with_config(config: DiffLTNConfig) -> Self {
        let tnorm = DiffTNorm::with_config(config.tnorm_config.clone());
        Self {
            config,
            tnorm,
            rules: Vec::new(),
            predicates: HashMap::new(),
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &DiffLTNConfig {
        &self.config
    }

    /// Get the t-norm operator.
    pub fn tnorm(&self) -> &DiffTNorm {
        &self.tnorm
    }

    // =========================================================================
    // Rule Management
    // =========================================================================

    /// Add a rule to the LTN.
    pub fn add_rule(&mut self, rule: LogicalRule) {
        self.rules.push(rule);
    }

    /// Add multiple rules.
    pub fn add_rules(&mut self, rules: impl IntoIterator<Item = LogicalRule>) {
        self.rules.extend(rules);
    }

    /// Get all rules.
    pub fn rules(&self) -> &[LogicalRule] {
        &self.rules
    }

    /// Get a rule by name.
    pub fn get_rule(&self, name: &str) -> Option<&LogicalRule> {
        self.rules.iter().find(|r| r.name == name)
    }

    /// Get a mutable rule by name.
    pub fn get_rule_mut(&mut self, name: &str) -> Option<&mut LogicalRule> {
        self.rules.iter_mut().find(|r| r.name == name)
    }

    /// Remove a rule by name.
    pub fn remove_rule(&mut self, name: &str) -> Option<LogicalRule> {
        self.rules
            .iter()
            .position(|r| r.name == name)
            .map(|i| self.rules.remove(i))
    }

    /// Set rule active status.
    pub fn set_rule_active(&mut self, name: &str, active: bool) -> bool {
        if let Some(rule) = self.get_rule_mut(name) {
            rule.active = active;
            true
        } else {
            false
        }
    }

    /// Get the number of active rules.
    pub fn num_active_rules(&self) -> usize {
        self.rules.iter().filter(|r| r.active).count()
    }

    // =========================================================================
    // Predicate Management
    // =========================================================================

    /// Register a learnable predicate.
    pub fn register_predicate(&mut self, name: &str, predicate: LearnablePredicate) {
        self.predicates
            .insert(name.to_string(), Arc::new(predicate));
    }

    /// Get a predicate by name.
    pub fn get_predicate(&self, name: &str) -> Option<&Arc<LearnablePredicate>> {
        self.predicates.get(name)
    }

    /// Apply predicates to create a grounding from raw inputs.
    pub fn apply_predicates(&self, inputs: &HashMap<String, Tensor>) -> Result<Grounding> {
        let mut grounding = Grounding::new();

        for (name, input) in inputs {
            if let Some(predicate) = self.predicates.get(name) {
                let truth_value = predicate.forward(input)?;
                grounding.set(name, truth_value);
            } else {
                // Pass through as-is if no predicate registered
                grounding.set(name, input.clone());
            }
        }

        Ok(grounding)
    }

    // =========================================================================
    // Satisfaction Computation
    // =========================================================================

    /// Evaluate a single rule's satisfaction.
    pub fn evaluate_rule(&self, rule: &LogicalRule, grounding: &Grounding) -> Result<Tensor> {
        rule.evaluate(grounding, &self.tnorm)
    }

    /// Evaluate all active rules and return individual satisfactions.
    pub fn evaluate_rules(&self, grounding: &Grounding) -> Result<HashMap<String, Tensor>> {
        let mut satisfactions = HashMap::new();

        for rule in &self.rules {
            if rule.active {
                let sat = self.evaluate_rule(rule, grounding)?;
                satisfactions.insert(rule.name.clone(), sat);
            }
        }

        Ok(satisfactions)
    }

    /// Compute the aggregated satisfaction score.
    ///
    /// Returns a scalar tensor representing overall rule satisfaction.
    pub fn satisfaction(&self, grounding: &Grounding) -> Result<Tensor> {
        let satisfactions = self.evaluate_rules(grounding)?;
        self.aggregate_satisfactions(&satisfactions)
    }

    /// Compute the satisfaction loss (1 - satisfaction).
    ///
    /// This is the primary loss function for training neural networks
    /// to satisfy logical constraints.
    pub fn satisfaction_loss(&self, grounding: &Grounding) -> Result<Tensor> {
        let satisfaction = self.satisfaction(grounding)?;
        let one = Tensor::ones_like(&satisfaction)?;
        one - satisfaction
    }

    /// Compute weighted satisfaction loss.
    ///
    /// Allows scaling the logical loss relative to other losses.
    pub fn weighted_loss(&self, grounding: &Grounding, weight: f32) -> Result<Tensor> {
        let loss = self.satisfaction_loss(grounding)?;
        loss.affine(weight as f64, 0.0)
    }

    /// Aggregate individual rule satisfactions into a single score.
    fn aggregate_satisfactions(&self, satisfactions: &HashMap<String, Tensor>) -> Result<Tensor> {
        if satisfactions.is_empty() {
            // No rules → perfect satisfaction
            return Tensor::new(1.0f32, &Device::Cpu);
        }

        // Collect satisfactions with weights
        let active_rules: Vec<_> = self
            .rules
            .iter()
            .filter(|r| r.active && satisfactions.contains_key(&r.name))
            .collect();

        if active_rules.is_empty() {
            return Tensor::new(1.0f32, &Device::Cpu);
        }

        let total_weight: f32 = active_rules.iter().map(|r| r.weight).sum();
        let norm_factor = if self.config.normalize_weights && total_weight > 0.0 {
            1.0 / total_weight
        } else {
            1.0
        };

        match self.config.aggregation {
            SatisfactionAggregation::Mean | SatisfactionAggregation::Sum => {
                let mut sum: Option<Tensor> = None;

                for rule in &active_rules {
                    let sat = satisfactions.get(&rule.name).unwrap();
                    let weight_val = if self.config.aggregation == SatisfactionAggregation::Mean {
                        rule.weight * norm_factor
                    } else {
                        rule.weight
                    };
                    let weighted = sat.affine(weight_val as f64, 0.0)?;

                    sum = Some(match sum {
                        Some(s) => (s + weighted)?,
                        None => weighted,
                    });
                }

                sum.ok_or_else(|| candle_core::Error::Msg("No satisfactions to aggregate".into()))
            }

            SatisfactionAggregation::Product => {
                // Weighted geometric mean
                let eps = 1e-7f32;
                let mut log_sum: Option<Tensor> = None;

                for rule in &active_rules {
                    let sat = satisfactions.get(&rule.name).unwrap();
                    // Add epsilon using affine: 1*x + eps
                    let safe_sat = sat.affine(1.0, eps as f64)?;
                    let log_sat = safe_sat.log()?;
                    let weighted = log_sat.affine((rule.weight * norm_factor) as f64, 0.0)?;

                    log_sum = Some(match log_sum {
                        Some(s) => (s + weighted)?,
                        None => weighted,
                    });
                }

                log_sum
                    .ok_or_else(|| candle_core::Error::Msg("No satisfactions to aggregate".into()))?
                    .exp()
            }

            SatisfactionAggregation::Min => {
                // Weighted soft minimum
                let temp = self.config.tnorm_config.temperature;
                let neg_temp = -1.0 / temp;

                let mut weighted_sum: Option<Tensor> = None;
                let mut weight_sum: Option<Tensor> = None;

                for rule in &active_rules {
                    let sat = satisfactions.get(&rule.name).unwrap();
                    let scaled = sat.affine(neg_temp as f64, 0.0)?;
                    let exp_scaled = scaled.exp()?;
                    let w = rule.weight * norm_factor;

                    let weighted_exp = exp_scaled.affine(w as f64, 0.0)?;

                    weighted_sum = Some(match weighted_sum {
                        Some(s) => (s + &weighted_exp)?,
                        None => weighted_exp.clone(),
                    });

                    let sat_weighted = (sat * &weighted_exp)?;
                    weight_sum = Some(match weight_sum {
                        Some(s) => (s + sat_weighted)?,
                        None => sat_weighted,
                    });
                }

                let w_sum = weighted_sum
                    .ok_or_else(|| candle_core::Error::Msg("No satisfactions".into()))?;
                let s_sum =
                    weight_sum.ok_or_else(|| candle_core::Error::Msg("No satisfactions".into()))?;

                s_sum / w_sum
            }
        }
    }

    // =========================================================================
    // Detailed Diagnostics
    // =========================================================================

    /// Get detailed satisfaction information for debugging/monitoring.
    pub fn satisfaction_details(&self, grounding: &Grounding) -> Result<SatisfactionReport> {
        let satisfactions = self.evaluate_rules(grounding)?;
        let overall = self.aggregate_satisfactions(&satisfactions)?;

        let rule_details: Vec<RuleSatisfaction> = self
            .rules
            .iter()
            .filter(|r| r.active)
            .map(|rule| {
                let sat = satisfactions
                    .get(&rule.name)
                    .map(|t| {
                        t.mean_all()
                            .and_then(|m| m.to_scalar::<f32>())
                            .unwrap_or(0.0)
                    })
                    .unwrap_or(0.0);
                RuleSatisfaction {
                    name: rule.name.clone(),
                    satisfaction: sat,
                    weight: rule.weight,
                    weighted_satisfaction: sat * rule.weight,
                }
            })
            .collect();

        let overall_scalar = overall.to_scalar::<f32>().unwrap_or(0.0);

        Ok(SatisfactionReport {
            overall_satisfaction: overall_scalar,
            overall_loss: 1.0 - overall_scalar,
            rule_satisfactions: rule_details,
            num_active_rules: self.num_active_rules(),
        })
    }
}

impl Default for DiffLTN {
    fn default() -> Self {
        Self::new(TNormType::Lukasiewicz)
    }
}

/// Report of satisfaction details for debugging/monitoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SatisfactionReport {
    /// Overall satisfaction score [0, 1]
    pub overall_satisfaction: f32,

    /// Overall loss (1 - satisfaction)
    pub overall_loss: f32,

    /// Per-rule satisfaction details
    pub rule_satisfactions: Vec<RuleSatisfaction>,

    /// Number of active rules
    pub num_active_rules: usize,
}

/// Satisfaction details for a single rule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleSatisfaction {
    /// Rule name
    pub name: String,

    /// Satisfaction score [0, 1]
    pub satisfaction: f32,

    /// Rule weight
    pub weight: f32,

    /// Weighted satisfaction (satisfaction * weight)
    pub weighted_satisfaction: f32,
}

// =============================================================================
// Pre-built Rule Sets
// =============================================================================

/// Pre-built rule sets for common use cases.
pub struct RuleSets;

impl RuleSets {
    /// Trading-specific rules for risk management.
    pub fn trading_rules() -> Vec<LogicalRule> {
        vec![
            // If confidence is high, allow action
            RuleBuilder::new("high_conf_allows_action")
                .implies("confidence_high", "action_allowed")
                .weight(1.0)
                .build(),
            // If risk is high, reduce position
            RuleBuilder::new("high_risk_reduces_position")
                .implies("risk_high", "reduce_position")
                .weight(1.5)
                .build(),
            // Position size must be valid
            RuleBuilder::new("position_valid")
                .requires("position_in_bounds")
                .weight(2.0)
                .build(),
            // If volatility is extreme, don't trade
            RuleBuilder::new("extreme_vol_no_trade")
                .formula(
                    Formula::var("volatility_extreme").implies(Formula::var("trade_action").not()),
                )
                .weight(1.5)
                .build(),
        ]
    }

    /// Consistency rules for multi-output networks.
    pub fn consistency_rules() -> Vec<LogicalRule> {
        vec![
            // Mutually exclusive actions: not (buy AND sell)
            RuleBuilder::new("buy_sell_exclusive")
                .formula(
                    Formula::var("action_buy")
                        .and(Formula::var("action_sell"))
                        .not(),
                )
                .weight(2.0)
                .build(),
            // At least one action must be taken: buy OR sell OR hold
            RuleBuilder::new("action_required")
                .formula(Formula::OrN(vec![
                    Formula::var("action_buy"),
                    Formula::var("action_sell"),
                    Formula::var("action_hold"),
                ]))
                .weight(1.0)
                .build(),
        ]
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_simple_grounding(device: &Device) -> Grounding {
        let mut g = Grounding::new();
        g.set("a", Tensor::new(0.8f32, device).unwrap());
        g.set("b", Tensor::new(0.6f32, device).unwrap());
        g.set("c", Tensor::new(0.9f32, device).unwrap());
        g
    }

    #[test]
    fn test_grounding_operations() {
        let device = Device::Cpu;
        let mut g = Grounding::new();

        g.set("x", Tensor::new(0.5f32, &device).unwrap());
        assert!(g.contains("x"));
        assert!(!g.contains("y"));
        assert_eq!(g.len(), 1);

        let x = g.get("x").unwrap();
        let val: f32 = x.to_scalar().unwrap();
        assert!((val - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_formula_variable_collection() {
        let f = Formula::var("a")
            .and(Formula::var("b"))
            .implies(Formula::var("c").or(Formula::var("a")));

        let vars = f.variables();
        assert_eq!(vars, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_formula_evaluation_simple() {
        let device = Device::Cpu;
        let grounding = create_simple_grounding(&device);
        let tnorm = DiffTNorm::new(TNormType::Lukasiewicz);

        // Test AND: a ∧ b = max(0, 0.8 + 0.6 - 1) = 0.4
        let f_and = Formula::var("a").and(Formula::var("b"));
        let result = f_and.evaluate(&grounding, &tnorm).unwrap();
        let val: f32 = result.to_scalar().unwrap();
        assert!((val - 0.4).abs() < 1e-5);

        // Test OR: a ∨ b = min(1, 0.8 + 0.6) = 1.0
        let f_or = Formula::var("a").or(Formula::var("b"));
        let result = f_or.evaluate(&grounding, &tnorm).unwrap();
        let val: f32 = result.to_scalar().unwrap();
        assert!((val - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_formula_implies() {
        let device = Device::Cpu;
        let grounding = create_simple_grounding(&device);
        let tnorm = DiffTNorm::new(TNormType::Lukasiewicz);

        // a → c = min(1, 1 - 0.8 + 0.9) = 1.0
        let f = Formula::var("a").implies(Formula::var("c"));
        let result = f.evaluate(&grounding, &tnorm).unwrap();
        let val: f32 = result.to_scalar().unwrap();
        assert!((val - 1.0).abs() < 1e-5);

        // a → b = min(1, 1 - 0.8 + 0.6) = 0.8
        let f = Formula::var("a").implies(Formula::var("b"));
        let result = f.evaluate(&grounding, &tnorm).unwrap();
        let val: f32 = result.to_scalar().unwrap();
        assert!((val - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_formula_not() {
        let device = Device::Cpu;
        let grounding = create_simple_grounding(&device);
        let tnorm = DiffTNorm::new(TNormType::Lukasiewicz);

        // ¬a = 1 - 0.8 = 0.2
        let f = Formula::var("a").not();
        let result = f.evaluate(&grounding, &tnorm).unwrap();
        let val: f32 = result.to_scalar().unwrap();
        assert!((val - 0.2).abs() < 1e-5);
    }

    #[test]
    fn test_formula_constants() {
        let device = Device::Cpu;
        let grounding = create_simple_grounding(&device);
        let tnorm = DiffTNorm::new(TNormType::Lukasiewicz);

        let f_true = Formula::true_const();
        let result = f_true.evaluate(&grounding, &tnorm).unwrap();
        let val: f32 = result.to_scalar().unwrap();
        assert!((val - 1.0).abs() < 1e-5);

        let f_false = Formula::false_const();
        let result = f_false.evaluate(&grounding, &tnorm).unwrap();
        let val: f32 = result.to_scalar().unwrap();
        assert!(val.abs() < 1e-5);
    }

    #[test]
    fn test_rule_builder() {
        let rule = RuleBuilder::new("test_rule")
            .implies("condition", "result")
            .weight(2.0)
            .build();

        assert_eq!(rule.name, "test_rule");
        assert_eq!(rule.weight, 2.0);
        assert!(rule.active);
    }

    #[test]
    fn test_ltn_add_rules() {
        let mut ltn = DiffLTN::new(TNormType::Lukasiewicz);

        ltn.add_rule(RuleBuilder::new("rule1").requires("a").build());
        ltn.add_rule(RuleBuilder::new("rule2").requires("b").build());

        assert_eq!(ltn.rules().len(), 2);
        assert!(ltn.get_rule("rule1").is_some());
        assert!(ltn.get_rule("nonexistent").is_none());
    }

    #[test]
    fn test_ltn_satisfaction() {
        let device = Device::Cpu;
        let mut ltn = DiffLTN::new(TNormType::Lukasiewicz);

        // Add rules that should be satisfied
        ltn.add_rule(RuleBuilder::new("high_a").requires("a").build());
        ltn.add_rule(RuleBuilder::new("high_c").requires("c").build());

        let grounding = create_simple_grounding(&device);
        let satisfaction = ltn.satisfaction(&grounding).unwrap();
        let val: f32 = satisfaction.to_scalar().unwrap();

        // Mean of (0.8, 0.9) = 0.85
        assert!((val - 0.85).abs() < 1e-5);
    }

    #[test]
    fn test_ltn_satisfaction_loss() {
        let device = Device::Cpu;
        let mut ltn = DiffLTN::new(TNormType::Lukasiewicz);

        ltn.add_rule(RuleBuilder::new("high_a").requires("a").build());

        let grounding = create_simple_grounding(&device);
        let loss = ltn.satisfaction_loss(&grounding).unwrap();
        let val: f32 = loss.to_scalar().unwrap();

        // Loss = 1 - 0.8 = 0.2
        assert!((val - 0.2).abs() < 1e-5);
    }

    #[test]
    fn test_ltn_weighted_rules() {
        let device = Device::Cpu;
        let mut ltn = DiffLTN::new(TNormType::Lukasiewicz);

        // Rule with weight 1.0, satisfaction 0.8
        ltn.add_rule(RuleBuilder::new("r1").requires("a").weight(1.0).build());
        // Rule with weight 3.0, satisfaction 0.6
        ltn.add_rule(RuleBuilder::new("r2").requires("b").weight(3.0).build());

        let grounding = create_simple_grounding(&device);
        let satisfaction = ltn.satisfaction(&grounding).unwrap();
        let val: f32 = satisfaction.to_scalar().unwrap();

        // Weighted mean: (1.0 * 0.8 + 3.0 * 0.6) / 4.0 = 2.6 / 4.0 = 0.65
        assert!((val - 0.65).abs() < 1e-5);
    }

    #[test]
    fn test_ltn_rule_deactivation() {
        let device = Device::Cpu;
        let mut ltn = DiffLTN::new(TNormType::Lukasiewicz);

        ltn.add_rule(RuleBuilder::new("r1").requires("a").build());
        ltn.add_rule(RuleBuilder::new("r2").requires("b").build());

        // Deactivate one rule
        ltn.set_rule_active("r2", false);

        assert_eq!(ltn.num_active_rules(), 1);

        let grounding = create_simple_grounding(&device);
        let satisfaction = ltn.satisfaction(&grounding).unwrap();
        let val: f32 = satisfaction.to_scalar().unwrap();

        // Only r1 active, satisfaction = 0.8
        assert!((val - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_ltn_batched_satisfaction() {
        let device = Device::Cpu;
        let mut ltn = DiffLTN::new(TNormType::Lukasiewicz);

        ltn.add_rule(
            RuleBuilder::new("r1")
                .requires("values")
                .forall("values")
                .build(),
        );

        let mut grounding = Grounding::new();
        grounding.set(
            "values",
            Tensor::new(&[0.9f32, 0.8, 0.7, 1.0], &device).unwrap(),
        );

        let satisfaction = ltn.satisfaction(&grounding).unwrap();
        let val: f32 = satisfaction.to_scalar().unwrap();

        // ForAll with Lukasiewicz uses mean: (0.9 + 0.8 + 0.7 + 1.0) / 4 = 0.85
        assert!((val - 0.85).abs() < 1e-5);
    }

    #[test]
    fn test_satisfaction_report() {
        let device = Device::Cpu;
        let mut ltn = DiffLTN::new(TNormType::Lukasiewicz);

        ltn.add_rule(RuleBuilder::new("r1").requires("a").weight(1.0).build());
        ltn.add_rule(RuleBuilder::new("r2").requires("b").weight(2.0).build());

        let grounding = create_simple_grounding(&device);
        let report = ltn.satisfaction_details(&grounding).unwrap();

        assert_eq!(report.num_active_rules, 2);
        assert_eq!(report.rule_satisfactions.len(), 2);

        let r1_detail = report
            .rule_satisfactions
            .iter()
            .find(|r| r.name == "r1")
            .unwrap();
        assert!((r1_detail.satisfaction - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_formula_quantifiers() {
        let device = Device::Cpu;
        let tnorm = DiffTNorm::new(TNormType::Lukasiewicz);

        let mut grounding = Grounding::new();
        grounding.set(
            "pred",
            Tensor::new(&[0.9f32, 0.8, 0.7, 1.0], &device).unwrap(),
        );

        // ForAll
        let f_forall = Formula::var("pred").forall(None);
        let result = f_forall.evaluate(&grounding, &tnorm).unwrap();
        let val: f32 = result.to_scalar().unwrap();
        assert!((val - 0.85).abs() < 1e-5);

        // Exists: 1 - mean(1 - values)
        let f_exists = Formula::var("pred").exists(None);
        let result = f_exists.evaluate(&grounding, &tnorm).unwrap();
        let val: f32 = result.to_scalar().unwrap();
        // 1 - mean(0.1, 0.2, 0.3, 0.0) = 1 - 0.15 = 0.85
        assert!((val - 0.85).abs() < 1e-5);
    }

    #[test]
    fn test_complex_formula() {
        let device = Device::Cpu;
        let tnorm = DiffTNorm::new(TNormType::Lukasiewicz);

        let mut grounding = Grounding::new();
        grounding.set("high_conf", Tensor::new(0.9f32, &device).unwrap());
        grounding.set("low_risk", Tensor::new(0.8f32, &device).unwrap());
        grounding.set("action", Tensor::new(0.7f32, &device).unwrap());

        // (high_conf ∧ low_risk) → action
        let formula = Formula::var("high_conf")
            .and(Formula::var("low_risk"))
            .implies(Formula::var("action"));

        let result = formula.evaluate(&grounding, &tnorm).unwrap();
        let val: f32 = result.to_scalar().unwrap();

        // high_conf ∧ low_risk = max(0, 0.9 + 0.8 - 1) = 0.7
        // 0.7 → 0.7 = min(1, 1 - 0.7 + 0.7) = 1.0
        assert!((val - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_prebuilt_rule_sets() {
        let trading_rules = RuleSets::trading_rules();
        assert!(!trading_rules.is_empty());

        let consistency_rules = RuleSets::consistency_rules();
        assert!(!consistency_rules.is_empty());
    }

    #[test]
    fn test_product_aggregation() {
        let device = Device::Cpu;
        let mut ltn = DiffLTN::with_config(DiffLTNConfig {
            aggregation: SatisfactionAggregation::Product,
            ..Default::default()
        });

        ltn.add_rule(RuleBuilder::new("r1").requires("a").build());
        ltn.add_rule(RuleBuilder::new("r2").requires("b").build());

        let grounding = create_simple_grounding(&device);
        let satisfaction = ltn.satisfaction(&grounding).unwrap();
        let val: f32 = satisfaction.to_scalar().unwrap();

        // Geometric mean of (0.8, 0.6) ≈ sqrt(0.48) ≈ 0.693
        let expected = (0.8f32 * 0.6).sqrt();
        assert!((val - expected).abs() < 0.01);
    }
}
