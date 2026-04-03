//! Trading Rule Predicates for the LTN Engine
//!
//! Part of the Prefrontal region
//! Component: ltn
//!
//! Provides a library of composable trading rule predicates that evaluate
//! market conditions and portfolio state, returning fuzzy truth values
//! in [0, 1] compatible with Łukasiewicz semantics.
//!
//! ## Features
//!
//! - **Position limit predicates**: Check absolute and relative position
//!   sizes against configurable limits with smooth fuzzy boundaries
//! - **Drawdown predicates**: Evaluate current drawdown against maximum
//!   allowed drawdown with proportional satisfaction curves
//! - **Concentration predicates**: Ensure portfolio diversification by
//!   limiting single-instrument concentration
//! - **Daily loss predicates**: Track intra-day P&L against daily loss
//!   limits with tiered severity
//! - **Time window predicates**: Restrict trading to specific time
//!   windows (market hours, session boundaries, blackout periods)
//! - **Volatility predicates**: Check implied/realised vol against
//!   acceptable ranges
//! - **Liquidity predicates**: Ensure sufficient market liquidity
//!   before order placement
//! - **Composition**: Combine predicates using Łukasiewicz AND, OR,
//!   NOT, IMPLICATION for complex rule construction
//! - **Predicate registry**: Named predicate collection with batch
//!   evaluation and scoring
//! - **Running statistics**: Per-predicate evaluation counts,
//!   satisfaction rates, violation history

use crate::common::{Error, Result};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Evaluation context
// ---------------------------------------------------------------------------

/// Market and portfolio state passed to predicates for evaluation.
#[derive(Debug, Clone)]
pub struct PredicateContext {
    /// Current instrument price.
    pub price: f64,
    /// Current trading volume.
    pub volume: f64,
    /// Current position size (signed: positive = long, negative = short).
    pub position_size: f64,
    /// Total portfolio value.
    pub portfolio_value: f64,
    /// Current drawdown as a fraction [0, 1].
    pub drawdown: f64,
    /// Intra-day realised P&L.
    pub daily_pnl: f64,
    /// Current hour of day (0–23).
    pub hour: u8,
    /// Current minute of hour (0–59).
    pub minute: u8,
    /// Realised volatility (annualised, as fraction).
    pub realised_vol: f64,
    /// Implied volatility (annualised, as fraction).
    pub implied_vol: f64,
    /// Bid-ask spread as a fraction of mid price.
    pub spread: f64,
    /// Average daily volume for the instrument.
    pub avg_daily_volume: f64,
    /// Number of open positions across portfolio.
    pub open_positions: usize,
    /// Maximum position value across all instruments.
    pub max_position_value: f64,
    /// Arbitrary named metrics for extensibility.
    pub metrics: HashMap<String, f64>,
}

impl Default for PredicateContext {
    fn default() -> Self {
        Self {
            price: 0.0,
            volume: 0.0,
            position_size: 0.0,
            portfolio_value: 1.0, // avoid division by zero
            drawdown: 0.0,
            daily_pnl: 0.0,
            hour: 12,
            minute: 0,
            realised_vol: 0.0,
            implied_vol: 0.0,
            spread: 0.0,
            avg_daily_volume: 1.0,
            open_positions: 0,
            max_position_value: 0.0,
            metrics: HashMap::new(),
        }
    }
}

impl PredicateContext {
    /// Create a new context with sensible defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: set price.
    pub fn with_price(mut self, price: f64) -> Self {
        self.price = price;
        self
    }

    /// Builder: set volume.
    pub fn with_volume(mut self, volume: f64) -> Self {
        self.volume = volume;
        self
    }

    /// Builder: set position size.
    pub fn with_position_size(mut self, size: f64) -> Self {
        self.position_size = size;
        self
    }

    /// Builder: set portfolio value.
    pub fn with_portfolio_value(mut self, value: f64) -> Self {
        self.portfolio_value = value;
        self
    }

    /// Builder: set drawdown.
    pub fn with_drawdown(mut self, dd: f64) -> Self {
        self.drawdown = dd;
        self
    }

    /// Builder: set daily P&L.
    pub fn with_daily_pnl(mut self, pnl: f64) -> Self {
        self.daily_pnl = pnl;
        self
    }

    /// Builder: set hour.
    pub fn with_hour(mut self, hour: u8) -> Self {
        self.hour = hour;
        self
    }

    /// Builder: set minute.
    pub fn with_minute(mut self, minute: u8) -> Self {
        self.minute = minute;
        self
    }

    /// Builder: set realised volatility.
    pub fn with_realised_vol(mut self, vol: f64) -> Self {
        self.realised_vol = vol;
        self
    }

    /// Builder: set implied volatility.
    pub fn with_implied_vol(mut self, vol: f64) -> Self {
        self.implied_vol = vol;
        self
    }

    /// Builder: set spread.
    pub fn with_spread(mut self, spread: f64) -> Self {
        self.spread = spread;
        self
    }

    /// Builder: set average daily volume.
    pub fn with_avg_daily_volume(mut self, adv: f64) -> Self {
        self.avg_daily_volume = adv;
        self
    }

    /// Builder: set open positions count.
    pub fn with_open_positions(mut self, n: usize) -> Self {
        self.open_positions = n;
        self
    }

    /// Builder: set max position value.
    pub fn with_max_position_value(mut self, v: f64) -> Self {
        self.max_position_value = v;
        self
    }

    /// Builder: set a named metric.
    pub fn with_metric(mut self, name: impl Into<String>, value: f64) -> Self {
        self.metrics.insert(name.into(), value);
        self
    }

    /// Get a named metric, returning 0.0 if not present.
    pub fn metric(&self, name: &str) -> f64 {
        self.metrics.get(name).copied().unwrap_or(0.0)
    }
}

// ---------------------------------------------------------------------------
// Predicate trait
// ---------------------------------------------------------------------------

/// A predicate that evaluates a trading condition and returns a fuzzy
/// truth value in [0, 1].
pub trait Predicate: Send + Sync {
    /// Evaluate the predicate against the given context.
    /// Returns a value in [0, 1] where 1.0 = fully satisfied.
    fn evaluate(&self, ctx: &PredicateContext) -> f64;

    /// Human-readable name for this predicate.
    fn name(&self) -> &str;

    /// Optional description.
    fn description(&self) -> &str {
        ""
    }
}

// ---------------------------------------------------------------------------
// Predicate record (for the registry)
// ---------------------------------------------------------------------------

/// Running statistics for a registered predicate.
#[derive(Debug, Clone)]
pub struct PredicateRecord {
    /// Name of the predicate.
    pub name: String,
    /// Total evaluations.
    pub eval_count: u64,
    /// Number of evaluations that returned < violation_threshold.
    pub violation_count: u64,
    /// Sum of truth values (for running mean).
    sum_truth: f64,
    /// Minimum truth observed.
    pub min_truth: f64,
    /// Maximum truth observed.
    pub max_truth: f64,
}

impl PredicateRecord {
    fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            eval_count: 0,
            violation_count: 0,
            sum_truth: 0.0,
            min_truth: 1.0,
            max_truth: 0.0,
        }
    }

    fn record(&mut self, truth: f64, violation_threshold: f64) {
        self.eval_count += 1;
        self.sum_truth += truth;
        if truth < self.min_truth {
            self.min_truth = truth;
        }
        if truth > self.max_truth {
            self.max_truth = truth;
        }
        if truth < violation_threshold {
            self.violation_count += 1;
        }
    }

    /// Running mean truth value.
    pub fn mean_truth(&self) -> f64 {
        if self.eval_count == 0 {
            return 1.0;
        }
        self.sum_truth / self.eval_count as f64
    }

    /// Violation rate.
    pub fn violation_rate(&self) -> f64 {
        if self.eval_count == 0 {
            return 0.0;
        }
        self.violation_count as f64 / self.eval_count as f64
    }
}

// ---------------------------------------------------------------------------
// Batch evaluation result
// ---------------------------------------------------------------------------

/// Result of evaluating all predicates in a registry.
#[derive(Debug, Clone)]
pub struct BatchResult {
    /// Per-predicate truth values (name → truth).
    pub truths: HashMap<String, f64>,
    /// Aggregate truth (Łukasiewicz AND of all predicates).
    pub aggregate: f64,
    /// Names of violated predicates (truth < threshold).
    pub violations: Vec<String>,
    /// Number of predicates evaluated.
    pub count: usize,
}

// ---------------------------------------------------------------------------
// Built-in predicates
// ---------------------------------------------------------------------------

/// Position size must be within an absolute limit.
pub struct PositionLimit {
    /// Maximum absolute position size.
    max_size: f64,
}

impl PositionLimit {
    pub fn new(max_size: f64) -> Self {
        Self { max_size }
    }
}

impl Predicate for PositionLimit {
    fn evaluate(&self, ctx: &PredicateContext) -> f64 {
        if self.max_size <= 0.0 {
            return 0.0;
        }
        let ratio = ctx.position_size.abs() / self.max_size;
        if ratio <= 1.0 {
            1.0 - ratio // fully satisfied when position is 0
        } else {
            0.0 // violated
        }
    }

    fn name(&self) -> &str {
        "position_limit"
    }

    fn description(&self) -> &str {
        "Position size within absolute limit"
    }
}

/// Position value must not exceed a fraction of portfolio value.
pub struct RelativePositionLimit {
    /// Maximum fraction of portfolio value [0, 1].
    max_fraction: f64,
}

impl RelativePositionLimit {
    pub fn new(max_fraction: f64) -> Self {
        Self { max_fraction }
    }
}

impl Predicate for RelativePositionLimit {
    fn evaluate(&self, ctx: &PredicateContext) -> f64 {
        if self.max_fraction <= 0.0 || ctx.portfolio_value <= 0.0 {
            return 0.0;
        }
        let position_value = ctx.position_size.abs() * ctx.price;
        let fraction = position_value / ctx.portfolio_value;
        if fraction <= self.max_fraction {
            1.0 - (fraction / self.max_fraction)
        } else {
            0.0
        }
    }

    fn name(&self) -> &str {
        "relative_position_limit"
    }

    fn description(&self) -> &str {
        "Position value within portfolio fraction limit"
    }
}

/// Drawdown must not exceed a maximum threshold.
pub struct MaxDrawdown {
    /// Maximum allowed drawdown as a fraction [0, 1].
    max_drawdown: f64,
}

impl MaxDrawdown {
    pub fn new(max_drawdown: f64) -> Self {
        Self { max_drawdown }
    }
}

impl Predicate for MaxDrawdown {
    fn evaluate(&self, ctx: &PredicateContext) -> f64 {
        if self.max_drawdown <= 0.0 {
            return if ctx.drawdown <= 0.0 { 1.0 } else { 0.0 };
        }
        let ratio = ctx.drawdown / self.max_drawdown;
        if ratio <= 1.0 { 1.0 - ratio } else { 0.0 }
    }

    fn name(&self) -> &str {
        "max_drawdown"
    }

    fn description(&self) -> &str {
        "Current drawdown within maximum allowed"
    }
}

/// Concentration: no single position may exceed a fraction of portfolio.
pub struct ConcentrationLimit {
    /// Maximum single-position fraction of portfolio [0, 1].
    max_concentration: f64,
}

impl ConcentrationLimit {
    pub fn new(max_concentration: f64) -> Self {
        Self { max_concentration }
    }
}

impl Predicate for ConcentrationLimit {
    fn evaluate(&self, ctx: &PredicateContext) -> f64 {
        if self.max_concentration <= 0.0 || ctx.portfolio_value <= 0.0 {
            return 0.0;
        }
        let concentration = ctx.max_position_value / ctx.portfolio_value;
        if concentration <= self.max_concentration {
            1.0 - (concentration / self.max_concentration)
        } else {
            0.0
        }
    }

    fn name(&self) -> &str {
        "concentration_limit"
    }

    fn description(&self) -> &str {
        "Portfolio concentration within limit"
    }
}

/// Daily loss must not exceed a maximum.
pub struct DailyLossLimit {
    /// Maximum daily loss as a positive value (absolute).
    max_loss: f64,
}

impl DailyLossLimit {
    pub fn new(max_loss: f64) -> Self {
        Self {
            max_loss: max_loss.abs(),
        }
    }
}

impl Predicate for DailyLossLimit {
    fn evaluate(&self, ctx: &PredicateContext) -> f64 {
        if self.max_loss <= 0.0 {
            return if ctx.daily_pnl >= 0.0 { 1.0 } else { 0.0 };
        }
        if ctx.daily_pnl >= 0.0 {
            return 1.0; // profitable day — fully compliant
        }
        let loss = ctx.daily_pnl.abs();
        let ratio = loss / self.max_loss;
        if ratio <= 1.0 { 1.0 - ratio } else { 0.0 }
    }

    fn name(&self) -> &str {
        "daily_loss_limit"
    }

    fn description(&self) -> &str {
        "Daily loss within maximum allowed"
    }
}

/// Trading must occur within specified market hours.
pub struct MarketHours {
    /// Start hour (inclusive, 24h format).
    start_hour: u8,
    /// End hour (exclusive, 24h format).
    end_hour: u8,
}

impl MarketHours {
    /// Create a market hours predicate.
    /// For NYSE-like hours: `MarketHours::new(9, 16)`.
    pub fn new(start_hour: u8, end_hour: u8) -> Self {
        Self {
            start_hour,
            end_hour,
        }
    }
}

impl Predicate for MarketHours {
    fn evaluate(&self, ctx: &PredicateContext) -> f64 {
        if self.start_hour <= self.end_hour {
            // Normal range (e.g. 9..16)
            if ctx.hour >= self.start_hour && ctx.hour < self.end_hour {
                1.0
            } else {
                0.0
            }
        } else {
            // Wrapping range (e.g. 22..6 for overnight)
            if ctx.hour >= self.start_hour || ctx.hour < self.end_hour {
                1.0
            } else {
                0.0
            }
        }
    }

    fn name(&self) -> &str {
        "market_hours"
    }

    fn description(&self) -> &str {
        "Trading within permitted market hours"
    }
}

/// Volatility must be within acceptable bounds.
pub struct VolatilityBounds {
    /// Minimum acceptable volatility (annualised fraction).
    min_vol: f64,
    /// Maximum acceptable volatility (annualised fraction).
    max_vol: f64,
    /// Whether to use implied (true) or realised (false) volatility.
    use_implied: bool,
}

impl VolatilityBounds {
    pub fn new(min_vol: f64, max_vol: f64, use_implied: bool) -> Self {
        Self {
            min_vol,
            max_vol,
            use_implied,
        }
    }
}

impl Predicate for VolatilityBounds {
    fn evaluate(&self, ctx: &PredicateContext) -> f64 {
        let vol = if self.use_implied {
            ctx.implied_vol
        } else {
            ctx.realised_vol
        };

        if vol < self.min_vol {
            // Below minimum — partially satisfied proportionally
            if self.min_vol <= 0.0 {
                1.0
            } else {
                (vol / self.min_vol).clamp(0.0, 1.0)
            }
        } else if vol > self.max_vol {
            // Above maximum — partially dissatisfied
            if self.max_vol <= 0.0 {
                0.0
            } else {
                (self.max_vol / vol).clamp(0.0, 1.0)
            }
        } else {
            1.0 // within bounds
        }
    }

    fn name(&self) -> &str {
        "volatility_bounds"
    }

    fn description(&self) -> &str {
        "Volatility within acceptable range"
    }
}

/// Sufficient liquidity: volume must be above a minimum and spread
/// must be below a maximum.
pub struct LiquidityCheck {
    /// Minimum volume as a fraction of average daily volume.
    min_volume_fraction: f64,
    /// Maximum bid-ask spread as a fraction of mid price.
    max_spread: f64,
}

impl LiquidityCheck {
    pub fn new(min_volume_fraction: f64, max_spread: f64) -> Self {
        Self {
            min_volume_fraction,
            max_spread,
        }
    }
}

impl Predicate for LiquidityCheck {
    fn evaluate(&self, ctx: &PredicateContext) -> f64 {
        // Volume check
        let volume_ratio = if ctx.avg_daily_volume > 0.0 {
            ctx.volume / ctx.avg_daily_volume
        } else {
            0.0
        };
        let volume_score = if volume_ratio >= self.min_volume_fraction {
            1.0
        } else if self.min_volume_fraction > 0.0 {
            volume_ratio / self.min_volume_fraction
        } else {
            1.0
        };

        // Spread check
        let spread_score = if ctx.spread <= self.max_spread {
            1.0
        } else if self.max_spread > 0.0 {
            (self.max_spread / ctx.spread).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Combine with Łukasiewicz AND
        (volume_score + spread_score - 1.0).max(0.0)
    }

    fn name(&self) -> &str {
        "liquidity_check"
    }

    fn description(&self) -> &str {
        "Sufficient market liquidity"
    }
}

/// Predicate based on a named metric with a threshold.
pub struct MetricThreshold {
    predicate_name: String,
    metric_name: String,
    threshold: f64,
    /// If true, metric must be >= threshold; if false, metric must be <= threshold.
    above: bool,
}

impl MetricThreshold {
    pub fn above(
        predicate_name: impl Into<String>,
        metric_name: impl Into<String>,
        threshold: f64,
    ) -> Self {
        Self {
            predicate_name: predicate_name.into(),
            metric_name: metric_name.into(),
            threshold,
            above: true,
        }
    }

    pub fn below(
        predicate_name: impl Into<String>,
        metric_name: impl Into<String>,
        threshold: f64,
    ) -> Self {
        Self {
            predicate_name: predicate_name.into(),
            metric_name: metric_name.into(),
            threshold,
            above: false,
        }
    }
}

impl Predicate for MetricThreshold {
    fn evaluate(&self, ctx: &PredicateContext) -> f64 {
        let value = ctx.metric(&self.metric_name);
        if self.above {
            if value >= self.threshold {
                1.0
            } else if self.threshold.abs() < f64::EPSILON {
                0.0
            } else {
                (value / self.threshold).clamp(0.0, 1.0)
            }
        } else if value <= self.threshold {
            1.0
        } else if self.threshold.abs() < f64::EPSILON {
            0.0
        } else {
            (self.threshold / value).clamp(0.0, 1.0)
        }
    }

    fn name(&self) -> &str {
        &self.predicate_name
    }
}

// ---------------------------------------------------------------------------
// Composition predicates
// ---------------------------------------------------------------------------

/// Łukasiewicz AND of two predicates.
pub struct And {
    a: Box<dyn Predicate>,
    b: Box<dyn Predicate>,
    predicate_name: String,
}

impl And {
    pub fn new(a: Box<dyn Predicate>, b: Box<dyn Predicate>) -> Self {
        let name = format!("({} AND {})", a.name(), b.name());
        Self {
            a,
            b,
            predicate_name: name,
        }
    }
}

impl Predicate for And {
    fn evaluate(&self, ctx: &PredicateContext) -> f64 {
        let va = self.a.evaluate(ctx);
        let vb = self.b.evaluate(ctx);
        (va + vb - 1.0).max(0.0)
    }

    fn name(&self) -> &str {
        &self.predicate_name
    }
}

/// Łukasiewicz OR of two predicates.
pub struct Or {
    a: Box<dyn Predicate>,
    b: Box<dyn Predicate>,
    predicate_name: String,
}

impl Or {
    pub fn new(a: Box<dyn Predicate>, b: Box<dyn Predicate>) -> Self {
        let name = format!("({} OR {})", a.name(), b.name());
        Self {
            a,
            b,
            predicate_name: name,
        }
    }
}

impl Predicate for Or {
    fn evaluate(&self, ctx: &PredicateContext) -> f64 {
        let va = self.a.evaluate(ctx);
        let vb = self.b.evaluate(ctx);
        (va + vb).min(1.0)
    }

    fn name(&self) -> &str {
        &self.predicate_name
    }
}

/// Łukasiewicz NOT of a predicate.
pub struct Not {
    inner: Box<dyn Predicate>,
    predicate_name: String,
}

impl Not {
    pub fn new(inner: Box<dyn Predicate>) -> Self {
        let name = format!("(NOT {})", inner.name());
        Self {
            inner,
            predicate_name: name,
        }
    }
}

impl Predicate for Not {
    fn evaluate(&self, ctx: &PredicateContext) -> f64 {
        1.0 - self.inner.evaluate(ctx)
    }

    fn name(&self) -> &str {
        &self.predicate_name
    }
}

/// Łukasiewicz IMPLICATION: A → B = min(1, 1 - A + B).
pub struct Implies {
    antecedent: Box<dyn Predicate>,
    consequent: Box<dyn Predicate>,
    predicate_name: String,
}

impl Implies {
    pub fn new(antecedent: Box<dyn Predicate>, consequent: Box<dyn Predicate>) -> Self {
        let name = format!("({} => {})", antecedent.name(), consequent.name());
        Self {
            antecedent,
            consequent,
            predicate_name: name,
        }
    }
}

impl Predicate for Implies {
    fn evaluate(&self, ctx: &PredicateContext) -> f64 {
        let a = self.antecedent.evaluate(ctx);
        let b = self.consequent.evaluate(ctx);
        (1.0 - a + b).min(1.0)
    }

    fn name(&self) -> &str {
        &self.predicate_name
    }
}

/// Constant truth value predicate (useful for testing and composition).
pub struct ConstTruth {
    value: f64,
    predicate_name: String,
}

impl ConstTruth {
    pub fn new(name: impl Into<String>, value: f64) -> Self {
        Self {
            value: value.clamp(0.0, 1.0),
            predicate_name: name.into(),
        }
    }
}

impl Predicate for ConstTruth {
    fn evaluate(&self, _ctx: &PredicateContext) -> f64 {
        self.value
    }

    fn name(&self) -> &str {
        &self.predicate_name
    }
}

// ---------------------------------------------------------------------------
// Predicate registry (Predicates struct)
// ---------------------------------------------------------------------------

/// A registry of named predicates with batch evaluation and statistics.
///
/// This is the main entry point for the ltn::predicates module. Register
/// predicates by name, evaluate them against a context, and inspect
/// per-predicate statistics.
pub struct Predicates {
    /// Registered predicates.
    predicates: Vec<(String, Box<dyn Predicate>)>,
    /// Per-predicate running statistics.
    records: HashMap<String, PredicateRecord>,
    /// Violation threshold.
    violation_threshold: f64,
    /// Maximum number of predicates.
    max_predicates: usize,
}

impl Default for Predicates {
    fn default() -> Self {
        Self::new()
    }
}

impl Predicates {
    /// Create a new predicate registry with default settings.
    pub fn new() -> Self {
        Self {
            predicates: Vec::new(),
            records: HashMap::new(),
            violation_threshold: 0.5,
            max_predicates: 256,
        }
    }

    /// Create with explicit configuration.
    pub fn with_config(violation_threshold: f64, max_predicates: usize) -> Result<Self> {
        if !(0.0..=1.0).contains(&violation_threshold) {
            return Err(Error::InvalidInput(
                "violation_threshold must be in [0, 1]".into(),
            ));
        }
        if max_predicates == 0 {
            return Err(Error::InvalidInput("max_predicates must be > 0".into()));
        }
        Ok(Self {
            predicates: Vec::new(),
            records: HashMap::new(),
            violation_threshold,
            max_predicates,
        })
    }

    /// Register a predicate.
    pub fn register(&mut self, predicate: Box<dyn Predicate>) -> Result<()> {
        let name = predicate.name().to_string();
        if self.records.contains_key(&name) {
            return Err(Error::InvalidInput(format!(
                "predicate '{}' already registered",
                name
            )));
        }
        if self.predicates.len() >= self.max_predicates {
            return Err(Error::ResourceExhausted(format!(
                "maximum predicates ({}) reached",
                self.max_predicates
            )));
        }
        self.records
            .insert(name.clone(), PredicateRecord::new(&name));
        self.predicates.push((name, predicate));
        Ok(())
    }

    /// Deregister a predicate by name.
    pub fn deregister(&mut self, name: &str) -> bool {
        let before = self.predicates.len();
        self.predicates.retain(|(n, _)| n != name);
        self.records.remove(name);
        self.predicates.len() < before
    }

    /// Number of registered predicates.
    pub fn count(&self) -> usize {
        self.predicates.len()
    }

    /// All predicate names in registration order.
    pub fn names(&self) -> Vec<String> {
        self.predicates.iter().map(|(n, _)| n.clone()).collect()
    }

    /// Get the statistics record for a predicate.
    pub fn record(&self, name: &str) -> Option<&PredicateRecord> {
        self.records.get(name)
    }

    /// Evaluate a single predicate by name.
    pub fn evaluate_one(&mut self, name: &str, ctx: &PredicateContext) -> Result<f64> {
        let pred = self
            .predicates
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, p)| p)
            .ok_or_else(|| Error::NotFound(format!("predicate '{}' not found", name)))?;
        let truth = pred.evaluate(ctx).clamp(0.0, 1.0);

        if let Some(rec) = self.records.get_mut(name) {
            rec.record(truth, self.violation_threshold);
        }

        Ok(truth)
    }

    /// Evaluate all registered predicates and return a batch result.
    pub fn evaluate_all(&mut self, ctx: &PredicateContext) -> BatchResult {
        let mut truths = HashMap::new();
        let mut values = Vec::new();
        let mut violations = Vec::new();

        for (name, pred) in &self.predicates {
            let truth = pred.evaluate(ctx).clamp(0.0, 1.0);
            truths.insert(name.clone(), truth);
            values.push(truth);

            if let Some(rec) = self.records.get_mut(name) {
                rec.record(truth, self.violation_threshold);
            }

            if truth < self.violation_threshold {
                violations.push(name.clone());
            }
        }

        // Aggregate using Łukasiewicz t-norm (AND)
        let aggregate = if values.is_empty() {
            1.0
        } else {
            values
                .iter()
                .copied()
                .fold(1.0, |acc, v| (acc + v - 1.0).max(0.0))
        };

        let count = self.predicates.len();

        BatchResult {
            truths,
            aggregate,
            violations,
            count,
        }
    }

    /// Main processing function (compatibility shim).
    pub fn process(&self) -> Result<()> {
        Ok(())
    }

    /// Reset all statistics.
    pub fn reset_stats(&mut self) {
        for rec in self.records.values_mut() {
            *rec = PredicateRecord::new(&rec.name);
        }
    }

    /// Clear all predicates and statistics.
    pub fn reset_all(&mut self) {
        self.predicates.clear();
        self.records.clear();
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn default_ctx() -> PredicateContext {
        PredicateContext::new()
            .with_price(100.0)
            .with_volume(1000.0)
            .with_position_size(10.0)
            .with_portfolio_value(100_000.0)
            .with_drawdown(0.02)
            .with_daily_pnl(500.0)
            .with_hour(10)
            .with_minute(30)
            .with_realised_vol(0.15)
            .with_implied_vol(0.18)
            .with_spread(0.001)
            .with_avg_daily_volume(10_000.0)
            .with_open_positions(3)
            .with_max_position_value(20_000.0)
    }

    // -----------------------------------------------------------------------
    // PredicateContext tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_context_defaults() {
        let ctx = PredicateContext::new();
        assert!((ctx.price - 0.0).abs() < 1e-10);
        assert!((ctx.portfolio_value - 1.0).abs() < 1e-10);
        assert_eq!(ctx.hour, 12);
    }

    #[test]
    fn test_context_builders() {
        let ctx = default_ctx();
        assert!((ctx.price - 100.0).abs() < 1e-10);
        assert!((ctx.volume - 1000.0).abs() < 1e-10);
        assert!((ctx.position_size - 10.0).abs() < 1e-10);
        assert_eq!(ctx.hour, 10);
        assert_eq!(ctx.minute, 30);
    }

    #[test]
    fn test_context_metric() {
        let ctx = PredicateContext::new().with_metric("sharpe", 1.5);
        assert!((ctx.metric("sharpe") - 1.5).abs() < 1e-10);
        assert!((ctx.metric("missing") - 0.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // PositionLimit tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_position_limit_satisfied() {
        let p = PositionLimit::new(100.0);
        let ctx = PredicateContext::new().with_position_size(50.0);
        let truth = p.evaluate(&ctx);
        // 1 - 50/100 = 0.5
        assert!((truth - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_position_limit_exact() {
        let p = PositionLimit::new(100.0);
        let ctx = PredicateContext::new().with_position_size(100.0);
        let truth = p.evaluate(&ctx);
        assert!((truth - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_position_limit_violated() {
        let p = PositionLimit::new(100.0);
        let ctx = PredicateContext::new().with_position_size(150.0);
        let truth = p.evaluate(&ctx);
        assert!((truth - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_position_limit_zero() {
        let p = PositionLimit::new(100.0);
        let ctx = PredicateContext::new().with_position_size(0.0);
        let truth = p.evaluate(&ctx);
        assert!((truth - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_position_limit_negative_position() {
        let p = PositionLimit::new(100.0);
        let ctx = PredicateContext::new().with_position_size(-50.0);
        let truth = p.evaluate(&ctx);
        assert!((truth - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_position_limit_zero_max() {
        let p = PositionLimit::new(0.0);
        let ctx = PredicateContext::new().with_position_size(10.0);
        assert!((p.evaluate(&ctx) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_position_limit_name() {
        let p = PositionLimit::new(100.0);
        assert_eq!(p.name(), "position_limit");
    }

    // -----------------------------------------------------------------------
    // RelativePositionLimit tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_relative_position_satisfied() {
        let p = RelativePositionLimit::new(0.1); // 10% of portfolio
        let ctx = PredicateContext::new()
            .with_position_size(5.0)
            .with_price(100.0)
            .with_portfolio_value(100_000.0);
        // position value = 500, fraction = 0.005, ratio = 0.005/0.1 = 0.05
        let truth = p.evaluate(&ctx);
        assert!((truth - 0.95).abs() < 1e-10);
    }

    #[test]
    fn test_relative_position_violated() {
        let p = RelativePositionLimit::new(0.1);
        let ctx = PredicateContext::new()
            .with_position_size(200.0)
            .with_price(100.0)
            .with_portfolio_value(100_000.0);
        // position value = 20000, fraction = 0.2, > 0.1 → violated
        let truth = p.evaluate(&ctx);
        assert!((truth - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_relative_position_name() {
        let p = RelativePositionLimit::new(0.1);
        assert_eq!(p.name(), "relative_position_limit");
    }

    // -----------------------------------------------------------------------
    // MaxDrawdown tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_drawdown_satisfied() {
        let p = MaxDrawdown::new(0.10);
        let ctx = PredicateContext::new().with_drawdown(0.03);
        // 1 - 0.03/0.10 = 0.7
        let truth = p.evaluate(&ctx);
        assert!((truth - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_drawdown_violated() {
        let p = MaxDrawdown::new(0.10);
        let ctx = PredicateContext::new().with_drawdown(0.15);
        assert!((p.evaluate(&ctx) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_drawdown_zero() {
        let p = MaxDrawdown::new(0.10);
        let ctx = PredicateContext::new().with_drawdown(0.0);
        assert!((p.evaluate(&ctx) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_drawdown_name() {
        let p = MaxDrawdown::new(0.10);
        assert_eq!(p.name(), "max_drawdown");
    }

    // -----------------------------------------------------------------------
    // ConcentrationLimit tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_concentration_satisfied() {
        let p = ConcentrationLimit::new(0.25);
        let ctx = PredicateContext::new()
            .with_portfolio_value(100_000.0)
            .with_max_position_value(10_000.0);
        // concentration = 0.1, ratio = 0.1/0.25 = 0.4, truth = 0.6
        let truth = p.evaluate(&ctx);
        assert!((truth - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_concentration_violated() {
        let p = ConcentrationLimit::new(0.25);
        let ctx = PredicateContext::new()
            .with_portfolio_value(100_000.0)
            .with_max_position_value(30_000.0);
        assert!((p.evaluate(&ctx) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_concentration_name() {
        let p = ConcentrationLimit::new(0.25);
        assert_eq!(p.name(), "concentration_limit");
    }

    // -----------------------------------------------------------------------
    // DailyLossLimit tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_daily_loss_profitable() {
        let p = DailyLossLimit::new(1000.0);
        let ctx = PredicateContext::new().with_daily_pnl(500.0);
        assert!((p.evaluate(&ctx) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_daily_loss_within_limit() {
        let p = DailyLossLimit::new(1000.0);
        let ctx = PredicateContext::new().with_daily_pnl(-300.0);
        // 1 - 300/1000 = 0.7
        assert!((p.evaluate(&ctx) - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_daily_loss_exceeded() {
        let p = DailyLossLimit::new(1000.0);
        let ctx = PredicateContext::new().with_daily_pnl(-1500.0);
        assert!((p.evaluate(&ctx) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_daily_loss_name() {
        let p = DailyLossLimit::new(1000.0);
        assert_eq!(p.name(), "daily_loss_limit");
    }

    // -----------------------------------------------------------------------
    // MarketHours tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_market_hours_within() {
        let p = MarketHours::new(9, 16);
        let ctx = PredicateContext::new().with_hour(10);
        assert!((p.evaluate(&ctx) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_market_hours_outside() {
        let p = MarketHours::new(9, 16);
        let ctx = PredicateContext::new().with_hour(20);
        assert!((p.evaluate(&ctx) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_market_hours_boundary_start() {
        let p = MarketHours::new(9, 16);
        let ctx = PredicateContext::new().with_hour(9);
        assert!((p.evaluate(&ctx) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_market_hours_boundary_end() {
        let p = MarketHours::new(9, 16);
        let ctx = PredicateContext::new().with_hour(16);
        assert!((p.evaluate(&ctx) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_market_hours_overnight() {
        let p = MarketHours::new(22, 6);
        let ctx_night = PredicateContext::new().with_hour(23);
        assert!((p.evaluate(&ctx_night) - 1.0).abs() < 1e-10);

        let ctx_early = PredicateContext::new().with_hour(3);
        assert!((p.evaluate(&ctx_early) - 1.0).abs() < 1e-10);

        let ctx_day = PredicateContext::new().with_hour(12);
        assert!((p.evaluate(&ctx_day) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_market_hours_name() {
        let p = MarketHours::new(9, 16);
        assert_eq!(p.name(), "market_hours");
    }

    // -----------------------------------------------------------------------
    // VolatilityBounds tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_vol_bounds_within() {
        let p = VolatilityBounds::new(0.10, 0.30, false);
        let ctx = PredicateContext::new().with_realised_vol(0.20);
        assert!((p.evaluate(&ctx) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_vol_bounds_too_low() {
        let p = VolatilityBounds::new(0.10, 0.30, false);
        let ctx = PredicateContext::new().with_realised_vol(0.05);
        // 0.05 / 0.10 = 0.5
        assert!((p.evaluate(&ctx) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_vol_bounds_too_high() {
        let p = VolatilityBounds::new(0.10, 0.30, false);
        let ctx = PredicateContext::new().with_realised_vol(0.60);
        // 0.30 / 0.60 = 0.5
        assert!((p.evaluate(&ctx) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_vol_bounds_implied() {
        let p = VolatilityBounds::new(0.10, 0.30, true);
        let ctx = PredicateContext::new().with_implied_vol(0.20);
        assert!((p.evaluate(&ctx) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_vol_bounds_name() {
        let p = VolatilityBounds::new(0.10, 0.30, false);
        assert_eq!(p.name(), "volatility_bounds");
    }

    // -----------------------------------------------------------------------
    // LiquidityCheck tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_liquidity_sufficient() {
        let p = LiquidityCheck::new(0.5, 0.005);
        let ctx = PredicateContext::new()
            .with_volume(8000.0)
            .with_avg_daily_volume(10_000.0)
            .with_spread(0.001);
        // volume_ratio = 0.8 >= 0.5 → score = 1.0
        // spread 0.001 <= 0.005 → score = 1.0
        // AND = max(0, 1.0 + 1.0 - 1.0) = 1.0
        assert!((p.evaluate(&ctx) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_liquidity_low_volume() {
        let p = LiquidityCheck::new(0.5, 0.005);
        let ctx = PredicateContext::new()
            .with_volume(2000.0)
            .with_avg_daily_volume(10_000.0)
            .with_spread(0.001);
        // volume_ratio = 0.2, volume_score = 0.2/0.5 = 0.4
        // spread_score = 1.0
        // AND = max(0, 0.4 + 1.0 - 1.0) = 0.4
        assert!((p.evaluate(&ctx) - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_liquidity_wide_spread() {
        let p = LiquidityCheck::new(0.5, 0.005);
        let ctx = PredicateContext::new()
            .with_volume(8000.0)
            .with_avg_daily_volume(10_000.0)
            .with_spread(0.010);
        // volume_score = 1.0
        // spread_score = 0.005/0.010 = 0.5
        // AND = max(0, 1.0 + 0.5 - 1.0) = 0.5
        assert!((p.evaluate(&ctx) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_liquidity_name() {
        let p = LiquidityCheck::new(0.5, 0.005);
        assert_eq!(p.name(), "liquidity_check");
    }

    // -----------------------------------------------------------------------
    // MetricThreshold tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_metric_above_satisfied() {
        let p = MetricThreshold::above("sharpe_ok", "sharpe", 1.0);
        let ctx = PredicateContext::new().with_metric("sharpe", 1.5);
        assert!((p.evaluate(&ctx) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_metric_above_partial() {
        let p = MetricThreshold::above("sharpe_ok", "sharpe", 2.0);
        let ctx = PredicateContext::new().with_metric("sharpe", 1.0);
        // 1.0 / 2.0 = 0.5
        assert!((p.evaluate(&ctx) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_metric_below_satisfied() {
        let p = MetricThreshold::below("var_ok", "var", 0.05);
        let ctx = PredicateContext::new().with_metric("var", 0.03);
        assert!((p.evaluate(&ctx) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_metric_below_partial() {
        let p = MetricThreshold::below("var_ok", "var", 0.05);
        let ctx = PredicateContext::new().with_metric("var", 0.10);
        // 0.05 / 0.10 = 0.5
        assert!((p.evaluate(&ctx) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_metric_name() {
        let p = MetricThreshold::above("my_metric", "x", 1.0);
        assert_eq!(p.name(), "my_metric");
    }

    // -----------------------------------------------------------------------
    // Composition tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_and_composition() {
        let a = Box::new(ConstTruth::new("a", 0.8));
        let b = Box::new(ConstTruth::new("b", 0.7));
        let p = And::new(a, b);
        let ctx = PredicateContext::new();
        // max(0, 0.8 + 0.7 - 1.0) = 0.5
        assert!((p.evaluate(&ctx) - 0.5).abs() < 1e-10);
        assert!(p.name().contains("AND"));
    }

    #[test]
    fn test_and_zero() {
        let a = Box::new(ConstTruth::new("a", 0.3));
        let b = Box::new(ConstTruth::new("b", 0.2));
        let p = And::new(a, b);
        let ctx = PredicateContext::new();
        // max(0, 0.3 + 0.2 - 1.0) = 0.0
        assert!((p.evaluate(&ctx) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_or_composition() {
        let a = Box::new(ConstTruth::new("a", 0.3));
        let b = Box::new(ConstTruth::new("b", 0.4));
        let p = Or::new(a, b);
        let ctx = PredicateContext::new();
        // min(1, 0.3 + 0.4) = 0.7
        assert!((p.evaluate(&ctx) - 0.7).abs() < 1e-10);
        assert!(p.name().contains("OR"));
    }

    #[test]
    fn test_or_saturates() {
        let a = Box::new(ConstTruth::new("a", 0.8));
        let b = Box::new(ConstTruth::new("b", 0.7));
        let p = Or::new(a, b);
        let ctx = PredicateContext::new();
        assert!((p.evaluate(&ctx) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_not_composition() {
        let inner = Box::new(ConstTruth::new("a", 0.3));
        let p = Not::new(inner);
        let ctx = PredicateContext::new();
        assert!((p.evaluate(&ctx) - 0.7).abs() < 1e-10);
        assert!(p.name().contains("NOT"));
    }

    #[test]
    fn test_implies_composition() {
        let a = Box::new(ConstTruth::new("a", 0.8));
        let b = Box::new(ConstTruth::new("b", 0.6));
        let p = Implies::new(a, b);
        let ctx = PredicateContext::new();
        // min(1, 1 - 0.8 + 0.6) = min(1, 0.8) = 0.8
        assert!((p.evaluate(&ctx) - 0.8).abs() < 1e-10);
        assert!(p.name().contains("=>"));
    }

    #[test]
    fn test_implies_false_premise() {
        let a = Box::new(ConstTruth::new("a", 0.0));
        let b = Box::new(ConstTruth::new("b", 0.3));
        let p = Implies::new(a, b);
        let ctx = PredicateContext::new();
        // min(1, 1 - 0 + 0.3) = 1.0
        assert!((p.evaluate(&ctx) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_const_truth() {
        let p = ConstTruth::new("const", 0.42);
        let ctx = PredicateContext::new();
        assert!((p.evaluate(&ctx) - 0.42).abs() < 1e-10);
        assert_eq!(p.name(), "const");
    }

    #[test]
    fn test_const_truth_clamped() {
        let p = ConstTruth::new("high", 1.5);
        let ctx = PredicateContext::new();
        assert!((p.evaluate(&ctx) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_nested_composition() {
        // (A AND B) OR (NOT C)
        let a = Box::new(ConstTruth::new("a", 0.9));
        let b = Box::new(ConstTruth::new("b", 0.8));
        let c = Box::new(ConstTruth::new("c", 0.3));

        let and_ab = Box::new(And::new(a, b)); // 0.7
        let not_c = Box::new(Not::new(c)); // 0.7
        let result = Or::new(and_ab, not_c); // min(1, 0.7 + 0.7) = 1.0

        let ctx = PredicateContext::new();
        assert!((result.evaluate(&ctx) - 1.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Predicates registry tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_registry_new() {
        let reg = Predicates::new();
        assert_eq!(reg.count(), 0);
    }

    #[test]
    fn test_registry_register() {
        let mut reg = Predicates::new();
        reg.register(Box::new(ConstTruth::new("test", 0.5)))
            .unwrap();
        assert_eq!(reg.count(), 1);
        assert_eq!(reg.names(), vec!["test"]);
    }

    #[test]
    fn test_registry_register_duplicate() {
        let mut reg = Predicates::new();
        reg.register(Box::new(ConstTruth::new("test", 0.5)))
            .unwrap();
        assert!(
            reg.register(Box::new(ConstTruth::new("test", 0.5)))
                .is_err()
        );
    }

    #[test]
    fn test_registry_max_capacity() {
        let mut reg = Predicates::with_config(0.5, 2).unwrap();
        reg.register(Box::new(ConstTruth::new("a", 0.5))).unwrap();
        reg.register(Box::new(ConstTruth::new("b", 0.5))).unwrap();
        assert!(reg.register(Box::new(ConstTruth::new("c", 0.5))).is_err());
    }

    #[test]
    fn test_registry_deregister() {
        let mut reg = Predicates::new();
        reg.register(Box::new(ConstTruth::new("test", 0.5)))
            .unwrap();
        assert!(reg.deregister("test"));
        assert_eq!(reg.count(), 0);
        assert!(!reg.deregister("test"));
    }

    #[test]
    fn test_registry_evaluate_one() {
        let mut reg = Predicates::new();
        reg.register(Box::new(ConstTruth::new("c", 0.7))).unwrap();
        let truth = reg.evaluate_one("c", &PredicateContext::new()).unwrap();
        assert!((truth - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_registry_evaluate_one_not_found() {
        let mut reg = Predicates::new();
        assert!(reg.evaluate_one("nope", &PredicateContext::new()).is_err());
    }

    #[test]
    fn test_registry_evaluate_all() {
        let mut reg = Predicates::new();
        reg.register(Box::new(ConstTruth::new("a", 0.9))).unwrap();
        reg.register(Box::new(ConstTruth::new("b", 0.8))).unwrap();

        let result = reg.evaluate_all(&PredicateContext::new());
        assert_eq!(result.count, 2);
        assert!((result.truths["a"] - 0.9).abs() < 1e-10);
        assert!((result.truths["b"] - 0.8).abs() < 1e-10);
        // aggregate = t_norm(0.9, 0.8) = 0.7
        assert!((result.aggregate - 0.7).abs() < 1e-10);
        assert!(result.violations.is_empty());
    }

    #[test]
    fn test_registry_evaluate_all_with_violations() {
        let mut reg = Predicates::new();
        reg.register(Box::new(ConstTruth::new("good", 0.9)))
            .unwrap();
        reg.register(Box::new(ConstTruth::new("bad", 0.3))).unwrap();

        let result = reg.evaluate_all(&PredicateContext::new());
        assert_eq!(result.violations.len(), 1);
        assert_eq!(result.violations[0], "bad");
    }

    #[test]
    fn test_registry_evaluate_all_empty() {
        let mut reg = Predicates::new();
        let result = reg.evaluate_all(&PredicateContext::new());
        assert!((result.aggregate - 1.0).abs() < 1e-10);
        assert_eq!(result.count, 0);
    }

    #[test]
    fn test_registry_record_stats() {
        let mut reg = Predicates::new();
        reg.register(Box::new(ConstTruth::new("r", 0.7))).unwrap();

        reg.evaluate_all(&PredicateContext::new());
        reg.evaluate_all(&PredicateContext::new());

        let rec = reg.record("r").unwrap();
        assert_eq!(rec.eval_count, 2);
        assert!((rec.mean_truth() - 0.7).abs() < 1e-10);
        assert_eq!(rec.violation_count, 0);
    }

    #[test]
    fn test_registry_record_violations() {
        let mut reg = Predicates::new();
        reg.register(Box::new(ConstTruth::new("low", 0.3))).unwrap();

        reg.evaluate_all(&PredicateContext::new());

        let rec = reg.record("low").unwrap();
        assert_eq!(rec.violation_count, 1);
        assert!((rec.violation_rate() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_registry_record_not_found() {
        let reg = Predicates::new();
        assert!(reg.record("nope").is_none());
    }

    #[test]
    fn test_registry_reset_stats() {
        let mut reg = Predicates::new();
        reg.register(Box::new(ConstTruth::new("r", 0.5))).unwrap();
        reg.evaluate_all(&PredicateContext::new());

        reg.reset_stats();

        let rec = reg.record("r").unwrap();
        assert_eq!(rec.eval_count, 0);
    }

    #[test]
    fn test_registry_reset_all() {
        let mut reg = Predicates::new();
        reg.register(Box::new(ConstTruth::new("r", 0.5))).unwrap();

        reg.reset_all();

        assert_eq!(reg.count(), 0);
    }

    #[test]
    fn test_registry_process() {
        let reg = Predicates::new();
        assert!(reg.process().is_ok());
    }

    // -----------------------------------------------------------------------
    // Config validation
    // -----------------------------------------------------------------------

    #[test]
    fn test_config_invalid_threshold() {
        assert!(Predicates::with_config(1.5, 10).is_err());
        assert!(Predicates::with_config(-0.1, 10).is_err());
    }

    #[test]
    fn test_config_invalid_max() {
        assert!(Predicates::with_config(0.5, 0).is_err());
    }

    // -----------------------------------------------------------------------
    // Integration: real predicates in registry
    // -----------------------------------------------------------------------

    #[test]
    fn test_registry_with_real_predicates() {
        let mut reg = Predicates::new();
        reg.register(Box::new(PositionLimit::new(100.0))).unwrap();
        reg.register(Box::new(MaxDrawdown::new(0.10))).unwrap();
        reg.register(Box::new(MarketHours::new(9, 16))).unwrap();

        let ctx = PredicateContext::new()
            .with_position_size(50.0)
            .with_drawdown(0.03)
            .with_hour(10);

        let result = reg.evaluate_all(&ctx);
        assert_eq!(result.count, 3);
        assert!(result.violations.is_empty());
        assert!(result.aggregate > 0.0);
    }

    #[test]
    fn test_registry_with_violations() {
        let mut reg = Predicates::new();
        reg.register(Box::new(PositionLimit::new(10.0))).unwrap();
        reg.register(Box::new(MarketHours::new(9, 16))).unwrap();

        let ctx = PredicateContext::new()
            .with_position_size(20.0) // violated
            .with_hour(20); // violated

        let result = reg.evaluate_all(&ctx);
        assert_eq!(result.violations.len(), 2);
        assert!((result.aggregate - 0.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // PredicateRecord edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_predicate_record_empty() {
        let rec = PredicateRecord::new("test");
        assert!((rec.mean_truth() - 1.0).abs() < 1e-10); // default
        assert!((rec.violation_rate() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_predicate_record_min_max() {
        let mut rec = PredicateRecord::new("test");
        rec.record(0.3, 0.5);
        rec.record(0.9, 0.5);
        rec.record(0.6, 0.5);

        assert!((rec.min_truth - 0.3).abs() < 1e-10);
        assert!((rec.max_truth - 0.9).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // De Morgan's law with composition predicates
    // -----------------------------------------------------------------------

    #[test]
    fn test_de_morgan_and_composition() {
        // NOT(A AND B) == (NOT A) OR (NOT B)
        let ctx = PredicateContext::new();

        let a_val = 0.7;
        let b_val = 0.6;

        // LHS: NOT(A AND B)
        let lhs = Not::new(Box::new(And::new(
            Box::new(ConstTruth::new("a", a_val)),
            Box::new(ConstTruth::new("b", b_val)),
        )));

        // RHS: (NOT A) OR (NOT B)
        let rhs = Or::new(
            Box::new(Not::new(Box::new(ConstTruth::new("a", a_val)))),
            Box::new(Not::new(Box::new(ConstTruth::new("b", b_val)))),
        );

        assert!((lhs.evaluate(&ctx) - rhs.evaluate(&ctx)).abs() < 1e-10);
    }

    #[test]
    fn test_de_morgan_or_composition() {
        // NOT(A OR B) == (NOT A) AND (NOT B)
        let ctx = PredicateContext::new();

        let a_val = 0.3;
        let b_val = 0.4;

        let lhs = Not::new(Box::new(Or::new(
            Box::new(ConstTruth::new("a", a_val)),
            Box::new(ConstTruth::new("b", b_val)),
        )));

        let rhs = And::new(
            Box::new(Not::new(Box::new(ConstTruth::new("a", a_val)))),
            Box::new(Not::new(Box::new(ConstTruth::new("b", b_val)))),
        );

        assert!((lhs.evaluate(&ctx) - rhs.evaluate(&ctx)).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Descriptions
    // -----------------------------------------------------------------------

    #[test]
    fn test_descriptions() {
        assert!(!PositionLimit::new(100.0).description().is_empty());
        assert!(!RelativePositionLimit::new(0.1).description().is_empty());
        assert!(!MaxDrawdown::new(0.1).description().is_empty());
        assert!(!ConcentrationLimit::new(0.25).description().is_empty());
        assert!(!DailyLossLimit::new(1000.0).description().is_empty());
        assert!(!MarketHours::new(9, 16).description().is_empty());
        assert!(
            !VolatilityBounds::new(0.1, 0.3, false)
                .description()
                .is_empty()
        );
        assert!(!LiquidityCheck::new(0.5, 0.005).description().is_empty());
    }
}
