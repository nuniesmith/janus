//! Lukasiewicz T-Norms — Fuzzy Logic Engine
//!
//! Part of the Prefrontal region
//! Component: ltn
//!
//! Implements continuous fuzzy logic operations based on Łukasiewicz
//! many-valued logic, the standard semantics used by Logic Tensor Networks.
//!
//! ## Features
//!
//! - **Core connectives**: AND (t-norm), OR (t-conorm), NOT (negation),
//!   IMPLICATION, BI-IMPLICATION (equivalence)
//! - **Hedges / modifiers**: `very` (concentration), `somewhat` (dilation),
//!   `slightly`, `extremely`, `more_or_less`
//! - **Quantifiers**: universal (`forall`), existential (`exists`),
//!   generalised mean quantifier
//! - **Aggregation**: min, max, arithmetic mean, weighted mean, product,
//!   p-mean (generalised)
//! - **EMA-smoothed aggregate truth**: Running truth-value signal with
//!   configurable decay for downstream consumers
//! - **Truth value wrapper**: Clamped `[0, 1]` newtype with arithmetic
//!   operator overloads
//! - **Evaluation context**: Collect named truth values, evaluate compound
//!   propositions, windowed diagnostics
//! - **Running statistics**: Total evaluations, mean / min / max truth,
//!   violation counts (truth < threshold)

use crate::common::{Error, Result};
use std::collections::{HashMap, VecDeque};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the fuzzy logic engine
#[derive(Debug, Clone)]
pub struct FuzzyLogicConfig {
    /// EMA decay for smoothed aggregate truth (0 < decay < 1)
    pub ema_decay: f64,
    /// Minimum evaluations before EMA is considered initialised
    pub min_samples: usize,
    /// Sliding window size for windowed diagnostics
    pub window_size: usize,
    /// Truth value below which a proposition is considered violated
    pub violation_threshold: f64,
    /// Maximum number of named propositions in the context
    pub max_propositions: usize,
}

impl Default for FuzzyLogicConfig {
    fn default() -> Self {
        Self {
            ema_decay: 0.1,
            min_samples: 5,
            window_size: 100,
            violation_threshold: 0.5,
            max_propositions: 256,
        }
    }
}

// ---------------------------------------------------------------------------
// Truth value
// ---------------------------------------------------------------------------

/// A truth value clamped to [0, 1].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Truth(f64);

impl Truth {
    /// Create a new truth value, clamping to [0, 1].
    pub fn new(v: f64) -> Self {
        Self(v.clamp(0.0, 1.0))
    }

    /// Absolutely true.
    pub fn one() -> Self {
        Self(1.0)
    }

    /// Absolutely false.
    pub fn zero() -> Self {
        Self(0.0)
    }

    /// Raw inner value.
    pub fn value(self) -> f64 {
        self.0
    }

    /// Whether this truth value satisfies the given threshold.
    pub fn satisfies(self, threshold: f64) -> bool {
        self.0 >= threshold
    }

    /// Whether this truth value is fully true (1.0).
    pub fn is_true(self) -> bool {
        (self.0 - 1.0).abs() < f64::EPSILON
    }

    /// Whether this truth value is fully false (0.0).
    pub fn is_false(self) -> bool {
        self.0.abs() < f64::EPSILON
    }
}

impl From<f64> for Truth {
    fn from(v: f64) -> Self {
        Self::new(v)
    }
}

impl From<Truth> for f64 {
    fn from(t: Truth) -> Self {
        t.0
    }
}

impl std::fmt::Display for Truth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.4}", self.0)
    }
}

// ---------------------------------------------------------------------------
// Core Łukasiewicz connectives
// ---------------------------------------------------------------------------

/// Łukasiewicz t-norm (fuzzy AND): max(0, a + b − 1)
pub fn t_norm(a: f64, b: f64) -> f64 {
    (a + b - 1.0).max(0.0)
}

/// Łukasiewicz t-conorm (fuzzy OR): min(1, a + b)
pub fn t_conorm(a: f64, b: f64) -> f64 {
    (a + b).min(1.0)
}

/// Łukasiewicz negation (fuzzy NOT): 1 − a
pub fn negation(a: f64) -> f64 {
    1.0 - a.clamp(0.0, 1.0)
}

/// Łukasiewicz implication: min(1, 1 − a + b)
pub fn implication(a: f64, b: f64) -> f64 {
    (1.0 - a + b).min(1.0)
}

/// Łukasiewicz bi-implication (equivalence): 1 − |a − b|
pub fn equivalence(a: f64, b: f64) -> f64 {
    1.0 - (a - b).abs()
}

/// N-ary t-norm: fold left with binary t-norm.
pub fn t_norm_n(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 1.0; // vacuous truth
    }
    values.iter().copied().fold(1.0, t_norm)
}

/// N-ary t-conorm: fold left with binary t-conorm.
pub fn t_conorm_n(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0; // vacuous falsity
    }
    values.iter().copied().fold(0.0, t_conorm)
}

// ---------------------------------------------------------------------------
// Hedges / linguistic modifiers
// ---------------------------------------------------------------------------

/// Concentration hedge ("very"): x²
pub fn very(x: f64) -> f64 {
    let v = x.clamp(0.0, 1.0);
    v * v
}

/// Dilation hedge ("somewhat"): √x
pub fn somewhat(x: f64) -> f64 {
    x.clamp(0.0, 1.0).sqrt()
}

/// "slightly": x − x² (peaks at 0.5)
pub fn slightly(x: f64) -> f64 {
    let v = x.clamp(0.0, 1.0);
    v - v * v
}

/// "extremely": x³
pub fn extremely(x: f64) -> f64 {
    let v = x.clamp(0.0, 1.0);
    v * v * v
}

/// "more or less": x^0.333
pub fn more_or_less(x: f64) -> f64 {
    x.clamp(0.0, 1.0).powf(1.0 / 3.0)
}

/// Apply a named hedge to a truth value.
pub fn apply_hedge(hedge: &Hedge, x: f64) -> f64 {
    match hedge {
        Hedge::Very => very(x),
        Hedge::Somewhat => somewhat(x),
        Hedge::Slightly => slightly(x),
        Hedge::Extremely => extremely(x),
        Hedge::MoreOrLess => more_or_less(x),
    }
}

/// Named hedge variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Hedge {
    Very,
    Somewhat,
    Slightly,
    Extremely,
    MoreOrLess,
}

// ---------------------------------------------------------------------------
// Quantifiers
// ---------------------------------------------------------------------------

/// Universal quantifier (∀): t-norm over all values.
pub fn forall(values: &[f64]) -> f64 {
    t_norm_n(values)
}

/// Existential quantifier (∃): t-conorm over all values.
pub fn exists(values: &[f64]) -> f64 {
    t_conorm_n(values)
}

/// Generalised mean quantifier: (1/n Σ x^p)^(1/p).
///
/// For `p = 1` this is the arithmetic mean.
/// As `p → ∞` it approaches the max.
/// As `p → -∞` it approaches the min.
pub fn gen_mean(values: &[f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    if p.abs() < f64::EPSILON {
        // Geometric mean
        let product: f64 = values.iter().map(|v| v.clamp(0.0, 1.0).ln()).sum();
        return (product / values.len() as f64).exp().clamp(0.0, 1.0);
    }
    let n = values.len() as f64;
    let s: f64 = values.iter().map(|v| v.clamp(0.0, 1.0).powf(p)).sum();
    (s / n).powf(1.0 / p).clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Aggregation helpers
// ---------------------------------------------------------------------------

/// Minimum of a slice.
pub fn agg_min(values: &[f64]) -> f64 {
    values
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min)
        .clamp(0.0, 1.0)
}

/// Maximum of a slice.
pub fn agg_max(values: &[f64]) -> f64 {
    values
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max)
        .clamp(0.0, 1.0)
}

/// Arithmetic mean.
pub fn agg_mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Weighted mean (weights need not sum to 1; they are normalised internally).
pub fn agg_weighted_mean(values: &[f64], weights: &[f64]) -> f64 {
    if values.is_empty() || weights.is_empty() {
        return 0.0;
    }
    let n = values.len().min(weights.len());
    let w_sum: f64 = weights[..n].iter().sum();
    if w_sum <= 0.0 {
        return 0.0;
    }
    let s: f64 = values[..n]
        .iter()
        .zip(weights[..n].iter())
        .map(|(v, w)| v * w)
        .sum();
    (s / w_sum).clamp(0.0, 1.0)
}

/// Product t-norm (probabilistic AND): Π xᵢ
pub fn agg_product(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 1.0;
    }
    values.iter().fold(1.0, |acc, v| acc * v.clamp(0.0, 1.0))
}

/// Named aggregation strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Aggregation {
    Min,
    Max,
    Mean,
    Product,
    TNorm,
    TConorm,
}

/// Apply a named aggregation to a slice.
pub fn aggregate(strategy: Aggregation, values: &[f64]) -> f64 {
    match strategy {
        Aggregation::Min => agg_min(values),
        Aggregation::Max => agg_max(values),
        Aggregation::Mean => agg_mean(values),
        Aggregation::Product => agg_product(values),
        Aggregation::TNorm => t_norm_n(values),
        Aggregation::TConorm => t_conorm_n(values),
    }
}

// ---------------------------------------------------------------------------
// Proposition
// ---------------------------------------------------------------------------

/// A named proposition with its current truth value.
#[derive(Debug, Clone)]
pub struct Proposition {
    /// Human-readable name.
    pub name: String,
    /// Current truth value.
    pub truth: Truth,
    /// Number of times this proposition has been evaluated.
    pub eval_count: u64,
    /// Sum of truth values (for running mean).
    sum_truth: f64,
    /// Minimum truth observed.
    pub min_truth: f64,
    /// Maximum truth observed.
    pub max_truth: f64,
}

impl Proposition {
    pub fn new(name: impl Into<String>, truth: f64) -> Self {
        let t = truth.clamp(0.0, 1.0);
        Self {
            name: name.into(),
            truth: Truth::new(t),
            eval_count: 1,
            sum_truth: t,
            min_truth: t,
            max_truth: t,
        }
    }

    /// Update the truth value with a new observation.
    pub fn update(&mut self, truth: f64) {
        let t = truth.clamp(0.0, 1.0);
        self.truth = Truth::new(t);
        self.eval_count += 1;
        self.sum_truth += t;
        if t < self.min_truth {
            self.min_truth = t;
        }
        if t > self.max_truth {
            self.max_truth = t;
        }
    }

    /// Running mean truth.
    pub fn mean_truth(&self) -> f64 {
        if self.eval_count == 0 {
            return 0.0;
        }
        self.sum_truth / self.eval_count as f64
    }
}

// ---------------------------------------------------------------------------
// Window record
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct WindowRecord {
    aggregate_truth: f64,
    violation_count: usize,
    proposition_count: usize,
    tick: u64,
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Running statistics for the fuzzy logic engine.
#[derive(Debug, Clone)]
pub struct FuzzyLogicStats {
    /// Total number of evaluation rounds.
    pub total_evaluations: u64,
    /// Total number of individual proposition violations across all rounds.
    pub total_violations: u64,
    /// Peak aggregate truth observed.
    pub peak_truth: f64,
    /// Lowest aggregate truth observed.
    pub trough_truth: f64,
    /// Sum of aggregate truth values (for mean).
    pub sum_truth: f64,
    /// Number of rounds where at least one violation occurred.
    pub rounds_with_violations: u64,
    /// Number of propositions currently registered.
    pub registered_propositions: usize,
}

impl Default for FuzzyLogicStats {
    fn default() -> Self {
        Self {
            total_evaluations: 0,
            total_violations: 0,
            peak_truth: 0.0,
            trough_truth: 1.0,
            sum_truth: 0.0,
            rounds_with_violations: 0,
            registered_propositions: 0,
        }
    }
}

impl FuzzyLogicStats {
    /// Mean aggregate truth over all rounds.
    pub fn mean_truth(&self) -> f64 {
        if self.total_evaluations == 0 {
            return 0.0;
        }
        self.sum_truth / self.total_evaluations as f64
    }

    /// Fraction of rounds with at least one violation.
    pub fn violation_rate(&self) -> f64 {
        if self.total_evaluations == 0 {
            return 0.0;
        }
        self.rounds_with_violations as f64 / self.total_evaluations as f64
    }
}

// ---------------------------------------------------------------------------
// Evaluation result
// ---------------------------------------------------------------------------

/// Result of a single evaluation round.
#[derive(Debug, Clone)]
pub struct EvaluationResult {
    /// Per-proposition truth values (name → truth).
    pub truths: HashMap<String, f64>,
    /// Aggregate truth (using the configured aggregation).
    pub aggregate: f64,
    /// EMA-smoothed aggregate truth.
    pub ema_truth: f64,
    /// Number of violations in this round.
    pub violations: Vec<String>,
}

// ---------------------------------------------------------------------------
// FuzzyLogic engine
// ---------------------------------------------------------------------------

/// Fuzzy logic evaluation engine using Łukasiewicz semantics.
///
/// Register named propositions, feed truth values, and the engine
/// maintains aggregate truth signals with EMA smoothing, violation
/// detection, and windowed diagnostics.
pub struct FuzzyLogic {
    config: FuzzyLogicConfig,

    /// Named propositions
    propositions: HashMap<String, Proposition>,

    /// Default aggregation strategy for computing the aggregate truth
    aggregation: Aggregation,

    /// EMA-smoothed aggregate truth
    ema_truth: f64,
    ema_initialized: bool,

    /// Sliding window for diagnostics
    recent: VecDeque<WindowRecord>,

    /// Current tick
    current_tick: u64,

    /// Running statistics
    stats: FuzzyLogicStats,
}

impl Default for FuzzyLogic {
    fn default() -> Self {
        Self::new()
    }
}

impl FuzzyLogic {
    /// Create a new fuzzy logic engine with default configuration.
    pub fn new() -> Self {
        Self::with_config(FuzzyLogicConfig::default()).unwrap()
    }

    /// Create with explicit configuration.
    pub fn with_config(config: FuzzyLogicConfig) -> Result<Self> {
        if config.ema_decay <= 0.0 || config.ema_decay >= 1.0 {
            return Err(Error::InvalidInput("ema_decay must be in (0, 1)".into()));
        }
        if config.window_size == 0 {
            return Err(Error::InvalidInput("window_size must be > 0".into()));
        }
        if config.max_propositions == 0 {
            return Err(Error::InvalidInput("max_propositions must be > 0".into()));
        }
        if config.violation_threshold < 0.0 || config.violation_threshold > 1.0 {
            return Err(Error::InvalidInput(
                "violation_threshold must be in [0, 1]".into(),
            ));
        }
        Ok(Self {
            config,
            propositions: HashMap::new(),
            aggregation: Aggregation::TNorm,
            ema_truth: 0.0,
            ema_initialized: false,
            recent: VecDeque::new(),
            current_tick: 0,
            stats: FuzzyLogicStats::default(),
        })
    }

    /// Set the aggregation strategy used for computing aggregate truth.
    pub fn set_aggregation(&mut self, strategy: Aggregation) {
        self.aggregation = strategy;
    }

    /// Get the current aggregation strategy.
    pub fn aggregation(&self) -> Aggregation {
        self.aggregation
    }

    // -----------------------------------------------------------------------
    // Proposition management
    // -----------------------------------------------------------------------

    /// Register a new proposition with an initial truth value.
    pub fn register(&mut self, name: impl Into<String>, truth: f64) -> Result<()> {
        let name = name.into();
        if self.propositions.contains_key(&name) {
            return Err(Error::InvalidInput(format!(
                "proposition '{}' already registered",
                name
            )));
        }
        if self.propositions.len() >= self.config.max_propositions {
            return Err(Error::ResourceExhausted(format!(
                "maximum propositions ({}) reached",
                self.config.max_propositions
            )));
        }
        self.propositions
            .insert(name.clone(), Proposition::new(name, truth));
        self.stats.registered_propositions = self.propositions.len();
        Ok(())
    }

    /// Remove a proposition by name.
    pub fn deregister(&mut self, name: &str) -> bool {
        let removed = self.propositions.remove(name).is_some();
        if removed {
            self.stats.registered_propositions = self.propositions.len();
        }
        removed
    }

    /// Update a single proposition's truth value.
    pub fn update(&mut self, name: &str, truth: f64) -> Result<()> {
        let prop = self
            .propositions
            .get_mut(name)
            .ok_or_else(|| Error::NotFound(format!("proposition '{}' not found", name)))?;
        prop.update(truth);
        Ok(())
    }

    /// Batch-update multiple propositions at once.
    pub fn update_batch(&mut self, updates: &[(&str, f64)]) -> Result<()> {
        for (name, truth) in updates {
            self.update(name, *truth)?;
        }
        Ok(())
    }

    /// Get a proposition by name.
    pub fn proposition(&self, name: &str) -> Option<&Proposition> {
        self.propositions.get(name)
    }

    /// Number of registered propositions.
    pub fn proposition_count(&self) -> usize {
        self.propositions.len()
    }

    /// All proposition names.
    pub fn proposition_names(&self) -> Vec<String> {
        let mut names: Vec<_> = self.propositions.keys().cloned().collect();
        names.sort();
        names
    }

    // -----------------------------------------------------------------------
    // Evaluation
    // -----------------------------------------------------------------------

    /// Evaluate all registered propositions and produce an aggregate truth.
    ///
    /// This is the main "tick" method. Call it after updating propositions
    /// to compute the aggregate truth, update EMA, and record diagnostics.
    pub fn evaluate(&mut self) -> EvaluationResult {
        self.current_tick += 1;

        // Collect current truths
        let mut truths = HashMap::new();
        let mut values = Vec::new();
        let mut violations = Vec::new();

        for (name, prop) in &self.propositions {
            let v = prop.truth.value();
            truths.insert(name.clone(), v);
            values.push(v);
            if v < self.config.violation_threshold {
                violations.push(name.clone());
            }
        }

        // Compute aggregate
        let agg = if values.is_empty() {
            1.0 // vacuous truth when no propositions
        } else {
            aggregate(self.aggregation, &values)
        };

        // Update EMA
        if !self.ema_initialized {
            self.ema_truth = agg;
            self.ema_initialized = true;
        } else {
            self.ema_truth =
                self.config.ema_decay * agg + (1.0 - self.config.ema_decay) * self.ema_truth;
        }

        // Update stats
        self.stats.total_evaluations += 1;
        self.stats.total_violations += violations.len() as u64;
        self.stats.sum_truth += agg;
        if agg > self.stats.peak_truth {
            self.stats.peak_truth = agg;
        }
        if agg < self.stats.trough_truth {
            self.stats.trough_truth = agg;
        }
        if !violations.is_empty() {
            self.stats.rounds_with_violations += 1;
        }

        // Window record
        let record = WindowRecord {
            aggregate_truth: agg,
            violation_count: violations.len(),
            proposition_count: self.propositions.len(),
            tick: self.current_tick,
        };
        self.recent.push_back(record);
        while self.recent.len() > self.config.window_size {
            self.recent.pop_front();
        }

        violations.sort();

        EvaluationResult {
            truths,
            aggregate: agg,
            ema_truth: self.ema_truth,
            violations,
        }
    }

    /// Evaluate a compound expression over the current proposition values.
    ///
    /// Supported syntax (prefix notation):
    /// - `(and A B)` — Łukasiewicz t-norm
    /// - `(or A B)` — Łukasiewicz t-conorm
    /// - `(not A)` — negation
    /// - `(imp A B)` — implication
    /// - `(eq A B)` — equivalence
    /// - `(very A)` — concentration hedge
    /// - `(somewhat A)` — dilation hedge
    /// - Plain name — look up proposition truth
    ///
    /// For simplicity, this evaluates flat binary/unary expressions
    /// (not recursive). Use the free functions for deeper nesting.
    pub fn eval_expr(&self, op: &str, args: &[&str]) -> Result<f64> {
        match op {
            "and" => {
                if args.len() < 2 {
                    return Err(Error::InvalidInput("and requires >= 2 args".into()));
                }
                let vals: Result<Vec<f64>> = args.iter().map(|a| self.resolve_name(a)).collect();
                Ok(t_norm_n(&vals?))
            }
            "or" => {
                if args.len() < 2 {
                    return Err(Error::InvalidInput("or requires >= 2 args".into()));
                }
                let vals: Result<Vec<f64>> = args.iter().map(|a| self.resolve_name(a)).collect();
                Ok(t_conorm_n(&vals?))
            }
            "not" => {
                if args.len() != 1 {
                    return Err(Error::InvalidInput("not requires exactly 1 arg".into()));
                }
                Ok(negation(self.resolve_name(args[0])?))
            }
            "imp" => {
                if args.len() != 2 {
                    return Err(Error::InvalidInput("imp requires exactly 2 args".into()));
                }
                Ok(implication(
                    self.resolve_name(args[0])?,
                    self.resolve_name(args[1])?,
                ))
            }
            "eq" => {
                if args.len() != 2 {
                    return Err(Error::InvalidInput("eq requires exactly 2 args".into()));
                }
                Ok(equivalence(
                    self.resolve_name(args[0])?,
                    self.resolve_name(args[1])?,
                ))
            }
            "very" => {
                if args.len() != 1 {
                    return Err(Error::InvalidInput("very requires exactly 1 arg".into()));
                }
                Ok(very(self.resolve_name(args[0])?))
            }
            "somewhat" => {
                if args.len() != 1 {
                    return Err(Error::InvalidInput(
                        "somewhat requires exactly 1 arg".into(),
                    ));
                }
                Ok(somewhat(self.resolve_name(args[0])?))
            }
            "extremely" => {
                if args.len() != 1 {
                    return Err(Error::InvalidInput(
                        "extremely requires exactly 1 arg".into(),
                    ));
                }
                Ok(extremely(self.resolve_name(args[0])?))
            }
            "forall" => {
                let vals: Result<Vec<f64>> = args.iter().map(|a| self.resolve_name(a)).collect();
                Ok(forall(&vals?))
            }
            "exists" => {
                let vals: Result<Vec<f64>> = args.iter().map(|a| self.resolve_name(a)).collect();
                Ok(exists(&vals?))
            }
            _ => Err(Error::InvalidInput(format!("unknown operator '{}'", op))),
        }
    }

    /// Resolve a name to its current truth value.
    fn resolve_name(&self, name: &str) -> Result<f64> {
        self.propositions
            .get(name)
            .map(|p| p.truth.value())
            .ok_or_else(|| Error::NotFound(format!("proposition '{}' not found", name)))
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Current EMA-smoothed aggregate truth.
    pub fn ema_truth(&self) -> f64 {
        self.ema_truth
    }

    /// Whether the EMA has had enough samples to be reliable.
    pub fn is_warmed_up(&self) -> bool {
        self.stats.total_evaluations >= self.config.min_samples as u64
    }

    /// Current tick count.
    pub fn current_tick(&self) -> u64 {
        self.current_tick
    }

    /// Running statistics.
    pub fn stats(&self) -> &FuzzyLogicStats {
        &self.stats
    }

    /// Confidence metric: increases as more evaluations are performed.
    pub fn confidence(&self) -> f64 {
        let n = self.stats.total_evaluations as f64;
        let min_s = self.config.min_samples as f64;
        (n / (n + min_s)).min(1.0)
    }

    // -----------------------------------------------------------------------
    // Windowed diagnostics
    // -----------------------------------------------------------------------

    /// Mean aggregate truth over the recent window.
    pub fn windowed_mean_truth(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.recent.iter().map(|r| r.aggregate_truth).sum();
        sum / self.recent.len() as f64
    }

    /// Fraction of recent rounds that contained at least one violation.
    pub fn windowed_violation_rate(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let count = self.recent.iter().filter(|r| r.violation_count > 0).count();
        count as f64 / self.recent.len() as f64
    }

    /// Minimum aggregate truth in the recent window.
    pub fn windowed_min_truth(&self) -> f64 {
        self.recent
            .iter()
            .map(|r| r.aggregate_truth)
            .fold(f64::INFINITY, f64::min)
            .max(0.0)
    }

    /// Maximum aggregate truth in the recent window.
    pub fn windowed_max_truth(&self) -> f64 {
        self.recent
            .iter()
            .map(|r| r.aggregate_truth)
            .fold(f64::NEG_INFINITY, f64::max)
            .min(1.0)
    }

    /// Detect whether truth is trending upward over the window.
    pub fn is_truth_increasing(&self) -> bool {
        if self.recent.len() < 4 {
            return false;
        }
        let mid = self.recent.len() / 2;
        let first_half: f64 = self
            .recent
            .iter()
            .take(mid)
            .map(|r| r.aggregate_truth)
            .sum::<f64>()
            / mid as f64;
        let second_half: f64 = self
            .recent
            .iter()
            .skip(mid)
            .map(|r| r.aggregate_truth)
            .sum::<f64>()
            / (self.recent.len() - mid) as f64;
        second_half > first_half + 0.01
    }

    /// Detect whether truth is trending downward over the window.
    pub fn is_truth_decreasing(&self) -> bool {
        if self.recent.len() < 4 {
            return false;
        }
        let mid = self.recent.len() / 2;
        let first_half: f64 = self
            .recent
            .iter()
            .take(mid)
            .map(|r| r.aggregate_truth)
            .sum::<f64>()
            / mid as f64;
        let second_half: f64 = self
            .recent
            .iter()
            .skip(mid)
            .map(|r| r.aggregate_truth)
            .sum::<f64>()
            / (self.recent.len() - mid) as f64;
        second_half < first_half - 0.01
    }

    // -----------------------------------------------------------------------
    // Reset
    // -----------------------------------------------------------------------

    /// Reset all internal state but keep propositions registered.
    pub fn reset(&mut self) {
        self.ema_truth = 0.0;
        self.ema_initialized = false;
        self.recent.clear();
        self.current_tick = 0;
        self.stats = FuzzyLogicStats {
            registered_propositions: self.propositions.len(),
            ..FuzzyLogicStats::default()
        };
        for prop in self.propositions.values_mut() {
            prop.truth = Truth::zero();
            prop.eval_count = 0;
            prop.sum_truth = 0.0;
            prop.min_truth = 1.0;
            prop.max_truth = 0.0;
        }
    }

    /// Reset everything including propositions.
    pub fn reset_all(&mut self) {
        self.propositions.clear();
        self.ema_truth = 0.0;
        self.ema_initialized = false;
        self.recent.clear();
        self.current_tick = 0;
        self.stats = FuzzyLogicStats::default();
    }

    /// Main processing function (compatibility shim).
    pub fn process(&self) -> Result<()> {
        Ok(())
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Truth value tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_truth_clamping() {
        assert_eq!(Truth::new(1.5).value(), 1.0);
        assert_eq!(Truth::new(-0.3).value(), 0.0);
        assert_eq!(Truth::new(0.7).value(), 0.7);
    }

    #[test]
    fn test_truth_constants() {
        assert!(Truth::one().is_true());
        assert!(Truth::zero().is_false());
        assert!(!Truth::one().is_false());
        assert!(!Truth::zero().is_true());
    }

    #[test]
    fn test_truth_satisfies() {
        let t = Truth::new(0.6);
        assert!(t.satisfies(0.5));
        assert!(!t.satisfies(0.7));
    }

    #[test]
    fn test_truth_display() {
        let t = Truth::new(0.5);
        let s = format!("{}", t);
        assert!(s.contains("0.5"));
    }

    #[test]
    fn test_truth_from_f64() {
        let t: Truth = 0.42.into();
        assert!((t.value() - 0.42).abs() < 1e-10);
    }

    #[test]
    fn test_truth_into_f64() {
        let t = Truth::new(0.88);
        let v: f64 = t.into();
        assert!((v - 0.88).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Core connective tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_t_norm_basic() {
        // max(0, 0.8 + 0.7 - 1) = max(0, 0.5) = 0.5
        assert!((t_norm(0.8, 0.7) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_t_norm_zeros() {
        // max(0, 0.3 + 0.2 - 1) = max(0, -0.5) = 0
        assert!((t_norm(0.3, 0.2) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_t_norm_ones() {
        assert!((t_norm(1.0, 1.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_t_norm_identity() {
        // t_norm(a, 1) = a
        assert!((t_norm(0.6, 1.0) - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_t_norm_commutativity() {
        assert!((t_norm(0.4, 0.9) - t_norm(0.9, 0.4)).abs() < 1e-10);
    }

    #[test]
    fn test_t_conorm_basic() {
        // min(1, 0.3 + 0.4) = 0.7
        assert!((t_conorm(0.3, 0.4) - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_t_conorm_saturates() {
        // min(1, 0.8 + 0.5) = 1.0
        assert!((t_conorm(0.8, 0.5) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_t_conorm_identity() {
        // t_conorm(a, 0) = a
        assert!((t_conorm(0.6, 0.0) - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_t_conorm_commutativity() {
        assert!((t_conorm(0.3, 0.7) - t_conorm(0.7, 0.3)).abs() < 1e-10);
    }

    #[test]
    fn test_negation() {
        assert!((negation(0.3) - 0.7).abs() < 1e-10);
        assert!((negation(1.0) - 0.0).abs() < 1e-10);
        assert!((negation(0.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_double_negation() {
        let x = 0.42;
        assert!((negation(negation(x)) - x).abs() < 1e-10);
    }

    #[test]
    fn test_implication_basic() {
        // min(1, 1 - 0.8 + 0.6) = min(1, 0.8) = 0.8
        assert!((implication(0.8, 0.6) - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_implication_true_premise_true_conclusion() {
        assert!((implication(1.0, 1.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_implication_false_premise() {
        // min(1, 1 - 0 + 0.3) = min(1, 1.3) = 1.0
        assert!((implication(0.0, 0.3) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_equivalence_same() {
        assert!((equivalence(0.7, 0.7) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_equivalence_different() {
        // 1 - |0.9 - 0.3| = 1 - 0.6 = 0.4
        assert!((equivalence(0.9, 0.3) - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_equivalence_symmetry() {
        assert!((equivalence(0.2, 0.8) - equivalence(0.8, 0.2)).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // N-ary connectives
    // -----------------------------------------------------------------------

    #[test]
    fn test_t_norm_n_empty() {
        assert!((t_norm_n(&[]) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_t_norm_n_single() {
        assert!((t_norm_n(&[0.7]) - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_t_norm_n_multiple() {
        // fold: t_norm(t_norm(1.0, 0.9), 0.8) = t_norm(0.9, 0.8) = 0.7
        let result = t_norm_n(&[0.9, 0.8]);
        assert!((result - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_t_conorm_n_empty() {
        assert!((t_conorm_n(&[]) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_t_conorm_n_single() {
        assert!((t_conorm_n(&[0.3]) - 0.3).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Hedge tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_very() {
        // 0.8^2 = 0.64
        assert!((very(0.8) - 0.64).abs() < 1e-10);
        assert!((very(1.0) - 1.0).abs() < 1e-10);
        assert!((very(0.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_somewhat() {
        // sqrt(0.64) = 0.8
        assert!((somewhat(0.64) - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_slightly() {
        // 0.5 - 0.25 = 0.25 (maximum at x=0.5)
        assert!((slightly(0.5) - 0.25).abs() < 1e-10);
        assert!((slightly(0.0) - 0.0).abs() < 1e-10);
        assert!((slightly(1.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_extremely() {
        // 0.5^3 = 0.125
        assert!((extremely(0.5) - 0.125).abs() < 1e-10);
    }

    #[test]
    fn test_more_or_less() {
        // 0.125^(1/3) = 0.5
        assert!((more_or_less(0.125) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_very_is_concentrating() {
        // very(x) <= x for x in [0, 1]
        for i in 0..=10 {
            let x = i as f64 / 10.0;
            assert!(very(x) <= x + 1e-10);
        }
    }

    #[test]
    fn test_somewhat_is_dilating() {
        // somewhat(x) >= x for x in [0, 1]
        for i in 0..=10 {
            let x = i as f64 / 10.0;
            assert!(somewhat(x) >= x - 1e-10);
        }
    }

    #[test]
    fn test_hedge_clamping() {
        // Out-of-range values are clamped
        assert!((very(2.0) - 1.0).abs() < 1e-10);
        assert!((somewhat(-1.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_apply_hedge_enum() {
        assert!((apply_hedge(&Hedge::Very, 0.8) - very(0.8)).abs() < 1e-10);
        assert!((apply_hedge(&Hedge::Somewhat, 0.64) - somewhat(0.64)).abs() < 1e-10);
        assert!((apply_hedge(&Hedge::Slightly, 0.5) - slightly(0.5)).abs() < 1e-10);
        assert!((apply_hedge(&Hedge::Extremely, 0.5) - extremely(0.5)).abs() < 1e-10);
        assert!((apply_hedge(&Hedge::MoreOrLess, 0.125) - more_or_less(0.125)).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Quantifier tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_forall_all_true() {
        assert!((forall(&[1.0, 1.0, 1.0]) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_forall_one_low() {
        // t_norm(t_norm(1.0, 1.0), 0.3) = t_norm(1.0, 0.3) = 0.3
        // then t_norm(0.3, 1.0) = 0.3
        let result = forall(&[1.0, 0.3, 1.0]);
        assert!((result - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_forall_empty() {
        assert!((forall(&[]) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_exists_all_false() {
        assert!((exists(&[0.0, 0.0, 0.0]) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_exists_one_high() {
        let result = exists(&[0.0, 0.9, 0.0]);
        assert!((result - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_exists_saturates() {
        let result = exists(&[0.6, 0.7]);
        assert!((result - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_gen_mean_arithmetic() {
        let result = gen_mean(&[0.2, 0.8], 1.0);
        assert!((result - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_gen_mean_empty() {
        assert!((gen_mean(&[], 1.0) - 0.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Aggregation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_agg_min() {
        assert!((agg_min(&[0.3, 0.7, 0.5]) - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_agg_max() {
        assert!((agg_max(&[0.3, 0.7, 0.5]) - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_agg_mean() {
        assert!((agg_mean(&[0.2, 0.4, 0.6]) - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_agg_mean_empty() {
        assert!((agg_mean(&[]) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_agg_weighted_mean() {
        // weighted mean of [0.2, 0.8] with weights [1, 3]
        // = (0.2*1 + 0.8*3) / (1+3) = 2.6/4 = 0.65
        assert!((agg_weighted_mean(&[0.2, 0.8], &[1.0, 3.0]) - 0.65).abs() < 1e-10);
    }

    #[test]
    fn test_agg_weighted_mean_empty() {
        assert!((agg_weighted_mean(&[], &[]) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_agg_weighted_mean_zero_weights() {
        assert!((agg_weighted_mean(&[0.5], &[0.0]) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_agg_product() {
        // 0.8 * 0.5 = 0.4
        assert!((agg_product(&[0.8, 0.5]) - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_agg_product_empty() {
        assert!((agg_product(&[]) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_aggregate_dispatch() {
        let vals = vec![0.3, 0.7, 0.5];
        assert!((aggregate(Aggregation::Min, &vals) - 0.3).abs() < 1e-10);
        assert!((aggregate(Aggregation::Max, &vals) - 0.7).abs() < 1e-10);
        assert!((aggregate(Aggregation::Mean, &vals) - 0.5).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Proposition tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_proposition_new() {
        let p = Proposition::new("test", 0.7);
        assert_eq!(p.name, "test");
        assert!((p.truth.value() - 0.7).abs() < 1e-10);
        assert_eq!(p.eval_count, 1);
    }

    #[test]
    fn test_proposition_update() {
        let mut p = Proposition::new("test", 0.5);
        p.update(0.8);
        assert!((p.truth.value() - 0.8).abs() < 1e-10);
        assert_eq!(p.eval_count, 2);
        assert!((p.min_truth - 0.5).abs() < 1e-10);
        assert!((p.max_truth - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_proposition_mean() {
        let mut p = Proposition::new("test", 0.4);
        p.update(0.6);
        // mean = (0.4 + 0.6) / 2 = 0.5
        assert!((p.mean_truth() - 0.5).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Engine construction & configuration
    // -----------------------------------------------------------------------

    #[test]
    fn test_default() {
        let fl = FuzzyLogic::new();
        assert_eq!(fl.proposition_count(), 0);
        assert_eq!(fl.current_tick(), 0);
    }

    #[test]
    fn test_invalid_config_ema_zero() {
        let mut cfg = FuzzyLogicConfig::default();
        cfg.ema_decay = 0.0;
        assert!(FuzzyLogic::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_ema_one() {
        let mut cfg = FuzzyLogicConfig::default();
        cfg.ema_decay = 1.0;
        assert!(FuzzyLogic::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_window_zero() {
        let mut cfg = FuzzyLogicConfig::default();
        cfg.window_size = 0;
        assert!(FuzzyLogic::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_max_propositions_zero() {
        let mut cfg = FuzzyLogicConfig::default();
        cfg.max_propositions = 0;
        assert!(FuzzyLogic::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_violation_threshold() {
        let mut cfg = FuzzyLogicConfig::default();
        cfg.violation_threshold = 1.5;
        assert!(FuzzyLogic::with_config(cfg).is_err());
    }

    // -----------------------------------------------------------------------
    // Registration
    // -----------------------------------------------------------------------

    #[test]
    fn test_register_proposition() {
        let mut fl = FuzzyLogic::new();
        assert!(fl.register("risk_ok", 0.9).is_ok());
        assert_eq!(fl.proposition_count(), 1);
        assert!(fl.proposition("risk_ok").is_some());
    }

    #[test]
    fn test_register_duplicate() {
        let mut fl = FuzzyLogic::new();
        fl.register("a", 0.5).unwrap();
        assert!(fl.register("a", 0.6).is_err());
    }

    #[test]
    fn test_register_max_capacity() {
        let mut cfg = FuzzyLogicConfig::default();
        cfg.max_propositions = 2;
        let mut fl = FuzzyLogic::with_config(cfg).unwrap();
        fl.register("a", 0.5).unwrap();
        fl.register("b", 0.5).unwrap();
        assert!(fl.register("c", 0.5).is_err());
    }

    #[test]
    fn test_deregister() {
        let mut fl = FuzzyLogic::new();
        fl.register("a", 0.5).unwrap();
        assert!(fl.deregister("a"));
        assert_eq!(fl.proposition_count(), 0);
        assert!(!fl.deregister("a")); // already removed
    }

    #[test]
    fn test_proposition_names_sorted() {
        let mut fl = FuzzyLogic::new();
        fl.register("z_prop", 0.5).unwrap();
        fl.register("a_prop", 0.5).unwrap();
        fl.register("m_prop", 0.5).unwrap();
        let names = fl.proposition_names();
        assert_eq!(names, vec!["a_prop", "m_prop", "z_prop"]);
    }

    // -----------------------------------------------------------------------
    // Update
    // -----------------------------------------------------------------------

    #[test]
    fn test_update() {
        let mut fl = FuzzyLogic::new();
        fl.register("a", 0.5).unwrap();
        fl.update("a", 0.9).unwrap();
        assert!((fl.proposition("a").unwrap().truth.value() - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_update_nonexistent() {
        let mut fl = FuzzyLogic::new();
        assert!(fl.update("nope", 0.5).is_err());
    }

    #[test]
    fn test_update_batch() {
        let mut fl = FuzzyLogic::new();
        fl.register("a", 0.0).unwrap();
        fl.register("b", 0.0).unwrap();
        fl.update_batch(&[("a", 0.8), ("b", 0.6)]).unwrap();
        assert!((fl.proposition("a").unwrap().truth.value() - 0.8).abs() < 1e-10);
        assert!((fl.proposition("b").unwrap().truth.value() - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_update_batch_partial_failure() {
        let mut fl = FuzzyLogic::new();
        fl.register("a", 0.0).unwrap();
        assert!(fl.update_batch(&[("a", 0.5), ("missing", 0.5)]).is_err());
    }

    // -----------------------------------------------------------------------
    // Evaluation
    // -----------------------------------------------------------------------

    #[test]
    fn test_evaluate_empty() {
        let mut fl = FuzzyLogic::new();
        let result = fl.evaluate();
        assert!((result.aggregate - 1.0).abs() < 1e-10); // vacuous truth
        assert_eq!(fl.current_tick(), 1);
    }

    #[test]
    fn test_evaluate_single_high() {
        let mut fl = FuzzyLogic::new();
        fl.register("p", 0.9).unwrap();
        let result = fl.evaluate();
        assert!((result.aggregate - 0.9).abs() < 1e-10);
        assert!(result.violations.is_empty());
    }

    #[test]
    fn test_evaluate_detects_violation() {
        let mut fl = FuzzyLogic::new();
        fl.register("risk_limit", 0.3).unwrap(); // below default threshold 0.5
        let result = fl.evaluate();
        assert_eq!(result.violations.len(), 1);
        assert_eq!(result.violations[0], "risk_limit");
    }

    #[test]
    fn test_evaluate_multiple() {
        let mut fl = FuzzyLogic::new();
        fl.set_aggregation(Aggregation::Mean);
        fl.register("a", 0.8).unwrap();
        fl.register("b", 0.6).unwrap();
        let result = fl.evaluate();
        assert!((result.aggregate - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_evaluate_tnorm_aggregate() {
        let mut fl = FuzzyLogic::new();
        fl.set_aggregation(Aggregation::TNorm);
        fl.register("a", 0.9).unwrap();
        fl.register("b", 0.8).unwrap();
        let result = fl.evaluate();
        // t_norm(0.9, 0.8) = 0.7
        assert!((result.aggregate - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_ema_initializes_on_first_eval() {
        let mut fl = FuzzyLogic::new();
        fl.register("p", 0.6).unwrap();
        let r = fl.evaluate();
        assert!((r.ema_truth - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_ema_smoothing() {
        let mut fl = FuzzyLogic::new();
        fl.register("p", 0.5).unwrap();
        fl.evaluate(); // ema = 0.5

        fl.update("p", 1.0).unwrap();
        let r2 = fl.evaluate();
        // ema = 0.1 * 1.0 + 0.9 * 0.5 = 0.55
        assert!((r2.ema_truth - 0.55).abs() < 1e-10);
    }

    #[test]
    fn test_stats_update() {
        let mut fl = FuzzyLogic::new();
        fl.register("p", 0.3).unwrap();
        fl.evaluate();

        let s = fl.stats();
        assert_eq!(s.total_evaluations, 1);
        assert_eq!(s.total_violations, 1);
        assert_eq!(s.rounds_with_violations, 1);
    }

    #[test]
    fn test_stats_mean_truth() {
        let mut fl = FuzzyLogic::new();
        fl.register("p", 0.4).unwrap();
        fl.evaluate();
        fl.update("p", 0.6).unwrap();
        fl.evaluate();

        // mean = (0.4 + 0.6) / 2 = 0.5
        assert!((fl.stats().mean_truth() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_stats_violation_rate() {
        let mut fl = FuzzyLogic::new();
        fl.register("p", 0.3).unwrap();
        fl.evaluate(); // violation
        fl.update("p", 0.8).unwrap();
        fl.evaluate(); // no violation

        // rate = 1/2 = 0.5
        assert!((fl.stats().violation_rate() - 0.5).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Expression evaluation
    // -----------------------------------------------------------------------

    #[test]
    fn test_eval_expr_and() {
        let mut fl = FuzzyLogic::new();
        fl.register("a", 0.9).unwrap();
        fl.register("b", 0.8).unwrap();
        let result = fl.eval_expr("and", &["a", "b"]).unwrap();
        assert!((result - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_eval_expr_or() {
        let mut fl = FuzzyLogic::new();
        fl.register("a", 0.3).unwrap();
        fl.register("b", 0.4).unwrap();
        let result = fl.eval_expr("or", &["a", "b"]).unwrap();
        assert!((result - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_eval_expr_not() {
        let mut fl = FuzzyLogic::new();
        fl.register("a", 0.3).unwrap();
        let result = fl.eval_expr("not", &["a"]).unwrap();
        assert!((result - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_eval_expr_imp() {
        let mut fl = FuzzyLogic::new();
        fl.register("a", 0.8).unwrap();
        fl.register("b", 0.6).unwrap();
        let result = fl.eval_expr("imp", &["a", "b"]).unwrap();
        assert!((result - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_eval_expr_eq() {
        let mut fl = FuzzyLogic::new();
        fl.register("a", 0.7).unwrap();
        fl.register("b", 0.7).unwrap();
        let result = fl.eval_expr("eq", &["a", "b"]).unwrap();
        assert!((result - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_eval_expr_very() {
        let mut fl = FuzzyLogic::new();
        fl.register("a", 0.8).unwrap();
        let result = fl.eval_expr("very", &["a"]).unwrap();
        assert!((result - 0.64).abs() < 1e-10);
    }

    #[test]
    fn test_eval_expr_somewhat() {
        let mut fl = FuzzyLogic::new();
        fl.register("a", 0.64).unwrap();
        let result = fl.eval_expr("somewhat", &["a"]).unwrap();
        assert!((result - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_eval_expr_extremely() {
        let mut fl = FuzzyLogic::new();
        fl.register("a", 0.5).unwrap();
        let result = fl.eval_expr("extremely", &["a"]).unwrap();
        assert!((result - 0.125).abs() < 1e-10);
    }

    #[test]
    fn test_eval_expr_forall() {
        let mut fl = FuzzyLogic::new();
        fl.register("a", 1.0).unwrap();
        fl.register("b", 0.3).unwrap();
        let result = fl.eval_expr("forall", &["a", "b"]).unwrap();
        assert!((result - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_eval_expr_exists() {
        let mut fl = FuzzyLogic::new();
        fl.register("a", 0.0).unwrap();
        fl.register("b", 0.7).unwrap();
        let result = fl.eval_expr("exists", &["a", "b"]).unwrap();
        assert!((result - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_eval_expr_unknown_op() {
        let fl = FuzzyLogic::new();
        assert!(fl.eval_expr("xor", &[]).is_err());
    }

    #[test]
    fn test_eval_expr_missing_prop() {
        let fl = FuzzyLogic::new();
        assert!(fl.eval_expr("not", &["nope"]).is_err());
    }

    #[test]
    fn test_eval_expr_and_insufficient_args() {
        let mut fl = FuzzyLogic::new();
        fl.register("a", 0.5).unwrap();
        assert!(fl.eval_expr("and", &["a"]).is_err());
    }

    #[test]
    fn test_eval_expr_not_too_many_args() {
        let mut fl = FuzzyLogic::new();
        fl.register("a", 0.5).unwrap();
        fl.register("b", 0.5).unwrap();
        assert!(fl.eval_expr("not", &["a", "b"]).is_err());
    }

    #[test]
    fn test_eval_expr_imp_wrong_arity() {
        let mut fl = FuzzyLogic::new();
        fl.register("a", 0.5).unwrap();
        assert!(fl.eval_expr("imp", &["a"]).is_err());
    }

    #[test]
    fn test_eval_expr_eq_wrong_arity() {
        let mut fl = FuzzyLogic::new();
        fl.register("a", 0.5).unwrap();
        assert!(fl.eval_expr("eq", &["a"]).is_err());
    }

    #[test]
    fn test_eval_expr_or_insufficient_args() {
        let mut fl = FuzzyLogic::new();
        fl.register("a", 0.5).unwrap();
        assert!(fl.eval_expr("or", &["a"]).is_err());
    }

    // -----------------------------------------------------------------------
    // Windowed diagnostics
    // -----------------------------------------------------------------------

    #[test]
    fn test_windowed_mean_empty() {
        let fl = FuzzyLogic::new();
        assert!((fl.windowed_mean_truth() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_windowed_mean_truth() {
        let mut fl = FuzzyLogic::new();
        fl.register("p", 0.4).unwrap();
        fl.evaluate();
        fl.update("p", 0.6).unwrap();
        fl.evaluate();

        // mean = (0.4 + 0.6) / 2 = 0.5
        assert!((fl.windowed_mean_truth() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_windowed_violation_rate() {
        let mut fl = FuzzyLogic::new();
        fl.register("p", 0.3).unwrap();
        fl.evaluate();
        fl.update("p", 0.8).unwrap();
        fl.evaluate();

        assert!((fl.windowed_violation_rate() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_windowed_violation_rate_empty() {
        let fl = FuzzyLogic::new();
        assert!((fl.windowed_violation_rate() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_windowed_min_max() {
        let mut fl = FuzzyLogic::new();
        fl.register("p", 0.3).unwrap();
        fl.evaluate();
        fl.update("p", 0.8).unwrap();
        fl.evaluate();

        assert!((fl.windowed_min_truth() - 0.3).abs() < 1e-10);
        assert!((fl.windowed_max_truth() - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_window_eviction() {
        let mut cfg = FuzzyLogicConfig::default();
        cfg.window_size = 3;
        let mut fl = FuzzyLogic::with_config(cfg).unwrap();
        fl.register("p", 0.1).unwrap();

        for _ in 0..5 {
            fl.evaluate();
        }

        // Window should only have 3 entries
        assert!(fl.recent.len() <= 3);
    }

    #[test]
    fn test_is_truth_increasing() {
        let mut fl = FuzzyLogic::new();
        fl.register("p", 0.0).unwrap();
        // Not enough data
        assert!(!fl.is_truth_increasing());

        // Generate increasing trend
        for i in 0..10 {
            fl.update("p", i as f64 / 10.0).unwrap();
            fl.evaluate();
        }
        assert!(fl.is_truth_increasing());
    }

    #[test]
    fn test_is_truth_decreasing() {
        let mut fl = FuzzyLogic::new();
        fl.register("p", 1.0).unwrap();

        // Generate decreasing trend
        for i in (0..10).rev() {
            fl.update("p", i as f64 / 10.0).unwrap();
            fl.evaluate();
        }
        assert!(fl.is_truth_decreasing());
    }

    #[test]
    fn test_trend_insufficient_data() {
        let mut fl = FuzzyLogic::new();
        fl.register("p", 0.5).unwrap();
        fl.evaluate();
        assert!(!fl.is_truth_increasing());
        assert!(!fl.is_truth_decreasing());
    }

    // -----------------------------------------------------------------------
    // Confidence & warmup
    // -----------------------------------------------------------------------

    #[test]
    fn test_is_warmed_up() {
        let mut fl = FuzzyLogic::new();
        fl.register("p", 0.5).unwrap();

        for _ in 0..4 {
            fl.evaluate();
        }
        assert!(!fl.is_warmed_up());

        fl.evaluate(); // 5th eval
        assert!(fl.is_warmed_up());
    }

    #[test]
    fn test_confidence_increases() {
        let mut fl = FuzzyLogic::new();
        fl.register("p", 0.5).unwrap();

        let c0 = fl.confidence();
        fl.evaluate();
        let c1 = fl.confidence();
        fl.evaluate();
        fl.evaluate();
        fl.evaluate();
        fl.evaluate();
        let c5 = fl.confidence();

        assert!(c1 > c0);
        assert!(c5 > c1);
        assert!(c5 <= 1.0);
    }

    // -----------------------------------------------------------------------
    // Aggregation strategy
    // -----------------------------------------------------------------------

    #[test]
    fn test_set_aggregation() {
        let mut fl = FuzzyLogic::new();
        fl.set_aggregation(Aggregation::Mean);
        assert_eq!(fl.aggregation(), Aggregation::Mean);
    }

    // -----------------------------------------------------------------------
    // Reset
    // -----------------------------------------------------------------------

    #[test]
    fn test_reset() {
        let mut fl = FuzzyLogic::new();
        fl.register("a", 0.8).unwrap();
        fl.register("b", 0.6).unwrap();
        fl.evaluate();
        fl.evaluate();

        fl.reset();

        assert_eq!(fl.current_tick(), 0);
        assert_eq!(fl.stats().total_evaluations, 0);
        assert_eq!(fl.proposition_count(), 2); // propositions kept
        assert!((fl.proposition("a").unwrap().truth.value() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_reset_all() {
        let mut fl = FuzzyLogic::new();
        fl.register("a", 0.8).unwrap();
        fl.evaluate();

        fl.reset_all();

        assert_eq!(fl.current_tick(), 0);
        assert_eq!(fl.stats().total_evaluations, 0);
        assert_eq!(fl.proposition_count(), 0);
    }

    // -----------------------------------------------------------------------
    // Process compatibility
    // -----------------------------------------------------------------------

    #[test]
    fn test_process() {
        let fl = FuzzyLogic::new();
        assert!(fl.process().is_ok());
    }

    // -----------------------------------------------------------------------
    // De Morgan's laws (Łukasiewicz)
    // -----------------------------------------------------------------------

    #[test]
    fn test_de_morgan_and() {
        // ¬(a ∧ b) = ¬a ∨ ¬b
        let a = 0.7;
        let b = 0.6;
        let lhs = negation(t_norm(a, b));
        let rhs = t_conorm(negation(a), negation(b));
        assert!((lhs - rhs).abs() < 1e-10);
    }

    #[test]
    fn test_de_morgan_or() {
        // ¬(a ∨ b) = ¬a ∧ ¬b
        let a = 0.3;
        let b = 0.4;
        let lhs = negation(t_conorm(a, b));
        let rhs = t_norm(negation(a), negation(b));
        assert!((lhs - rhs).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Contraposition
    // -----------------------------------------------------------------------

    #[test]
    fn test_contraposition() {
        // a → b ≡ ¬b → ¬a  (holds in Łukasiewicz logic)
        let a = 0.6;
        let b = 0.8;
        let lhs = implication(a, b);
        let rhs = implication(negation(b), negation(a));
        assert!((lhs - rhs).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Stats defaults
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_defaults() {
        let s = FuzzyLogicStats::default();
        assert_eq!(s.total_evaluations, 0);
        assert_eq!(s.total_violations, 0);
        assert!((s.mean_truth() - 0.0).abs() < 1e-10);
        assert!((s.violation_rate() - 0.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Peak & trough tracking
    // -----------------------------------------------------------------------

    #[test]
    fn test_peak_and_trough_tracking() {
        let mut fl = FuzzyLogic::new();
        fl.register("p", 0.5).unwrap();
        fl.evaluate();

        fl.update("p", 0.9).unwrap();
        fl.evaluate();

        fl.update("p", 0.2).unwrap();
        fl.evaluate();

        assert!((fl.stats().peak_truth - 0.9).abs() < 1e-10);
        assert!((fl.stats().trough_truth - 0.2).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Edge: t-norm annihilator
    // -----------------------------------------------------------------------

    #[test]
    fn test_t_norm_annihilator() {
        // t_norm(a, 0) = 0 for any a
        assert!((t_norm(0.999, 0.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_t_conorm_annihilator() {
        // t_conorm(a, 1) = 1 for any a
        assert!((t_conorm(0.001, 1.0) - 1.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Multiple violations in one round
    // -----------------------------------------------------------------------

    #[test]
    fn test_multiple_violations() {
        let mut fl = FuzzyLogic::new();
        fl.register("a", 0.1).unwrap();
        fl.register("b", 0.2).unwrap();
        fl.register("c", 0.9).unwrap();

        let result = fl.evaluate();
        assert_eq!(result.violations.len(), 2);
        assert!(result.violations.contains(&"a".to_string()));
        assert!(result.violations.contains(&"b".to_string()));
    }

    // -----------------------------------------------------------------------
    // Eval result contains all propositions
    // -----------------------------------------------------------------------

    #[test]
    fn test_eval_result_contains_all_props() {
        let mut fl = FuzzyLogic::new();
        fl.register("x", 0.3).unwrap();
        fl.register("y", 0.7).unwrap();

        let result = fl.evaluate();
        assert_eq!(result.truths.len(), 2);
        assert!(result.truths.contains_key("x"));
        assert!(result.truths.contains_key("y"));
    }

    // -----------------------------------------------------------------------
    // N-ary eval_expr (and/or with >2 args)
    // -----------------------------------------------------------------------

    #[test]
    fn test_eval_expr_and_three_args() {
        let mut fl = FuzzyLogic::new();
        fl.register("a", 1.0).unwrap();
        fl.register("b", 0.9).unwrap();
        fl.register("c", 0.8).unwrap();
        let result = fl.eval_expr("and", &["a", "b", "c"]).unwrap();
        // t_norm_n([1.0, 0.9, 0.8]) = t_norm(t_norm(1.0, 0.9), 0.8) = t_norm(0.9, 0.8) = 0.7
        // Actually: fold starts with 1.0. t_norm(1.0, 1.0)=1.0, t_norm(1.0, 0.9)=0.9, t_norm(0.9, 0.8)=0.7
        assert!((result - 0.7).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // EMA after reset
    // -----------------------------------------------------------------------

    #[test]
    fn test_ema_reinitializes_after_reset() {
        let mut fl = FuzzyLogic::new();
        fl.register("p", 0.5).unwrap();
        fl.evaluate();
        fl.evaluate();

        fl.reset();
        fl.update("p", 0.9).unwrap();
        let r = fl.evaluate();
        // Should reinitialize EMA to 0.9
        assert!((r.ema_truth - 0.9).abs() < 1e-10);
    }
}
