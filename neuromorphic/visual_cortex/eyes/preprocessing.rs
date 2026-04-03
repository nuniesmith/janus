//! Data normalization and cleaning
//!
//! Part of the Visual Cortex region
//! Component: eyes
//!
//! Provides a data normalisation and cleaning pipeline for raw market data
//! before it is fed into GAF encoding or pattern recognition. Supports
//! multiple normalisation strategies, outlier detection and clamping,
//! missing-value imputation, sliding z-score computation, and per-field
//! running statistics.
//!
//! ## Features
//!
//! - **Multiple normalisation strategies**: Min-max, z-score, robust
//!   (median-based), log-transform, and percentage-change
//! - **Outlier detection**: Sliding-window z-score–based outlier flagging
//!   with configurable threshold and optional clamping
//! - **Missing-value imputation**: Forward-fill, mean, median, or zero
//!   imputation for gaps in incoming data
//! - **Running statistics**: Online mean, variance, min, max per field
//!   using Welford's algorithm
//! - **Windowed statistics**: Sliding-window mean and standard deviation
//!   for adaptive normalisation
//! - **Pipeline composition**: Chain multiple preprocessing steps and
//!   apply them in order
//! - **NaN / Inf filtering**: Automatically rejects or replaces
//!   non-finite values

use crate::common::{Error, Result};
use std::collections::{HashMap, VecDeque};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the preprocessing engine.
#[derive(Debug, Clone)]
pub struct PreprocessingConfig {
    /// Maximum number of named fields that can be registered.
    pub max_fields: usize,
    /// Default normalisation strategy for new fields.
    pub default_strategy: NormStrategy,
    /// Sliding window size for windowed statistics and outlier detection.
    pub window_size: usize,
    /// Z-score threshold for outlier detection.
    pub outlier_z_threshold: f64,
    /// Whether to clamp outliers to the threshold boundary (vs reject).
    pub clamp_outliers: bool,
    /// Default imputation strategy for missing values.
    pub default_imputation: ImputeStrategy,
    /// Minimum observations before normalisation is applied.
    pub min_observations: usize,
    /// Whether to replace NaN/Inf with imputed values automatically.
    pub filter_non_finite: bool,
}

impl Default for PreprocessingConfig {
    fn default() -> Self {
        Self {
            max_fields: 128,
            default_strategy: NormStrategy::None,
            window_size: 100,
            outlier_z_threshold: 3.0,
            clamp_outliers: true,
            default_imputation: ImputeStrategy::ForwardFill,
            min_observations: 10,
            filter_non_finite: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Normalisation strategies
// ---------------------------------------------------------------------------

/// Normalisation strategy for a field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormStrategy {
    /// No normalisation — pass through raw values.
    None,
    /// Min-max scaling to [0, 1].
    MinMax,
    /// Z-score standardisation (subtract mean, divide by std dev).
    ZScore,
    /// Robust scaling using median and IQR (approximated via windowed
    /// percentiles). Falls back to z-score when insufficient data.
    Robust,
    /// Natural logarithm transform (only valid for positive values).
    Log,
    /// Percentage change from previous value.
    PctChange,
}

impl Default for NormStrategy {
    fn default() -> Self {
        Self::None
    }
}

// ---------------------------------------------------------------------------
// Imputation strategies
// ---------------------------------------------------------------------------

/// Strategy for imputing missing values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImputeStrategy {
    /// Use the last known value.
    ForwardFill,
    /// Use the running mean.
    Mean,
    /// Use the windowed median (approximated).
    Median,
    /// Replace with zero.
    Zero,
    /// Do not impute — leave as NaN / reject.
    Skip,
}

impl Default for ImputeStrategy {
    fn default() -> Self {
        Self::ForwardFill
    }
}

// ---------------------------------------------------------------------------
// Field specification
// ---------------------------------------------------------------------------

/// Per-field preprocessing specification.
#[derive(Debug, Clone)]
pub struct FieldConfig {
    /// Field name.
    pub name: String,
    /// Normalisation strategy.
    pub norm: NormStrategy,
    /// Imputation strategy.
    pub impute: ImputeStrategy,
    /// Whether outlier detection is enabled.
    pub outlier_detection: bool,
    /// Custom z-score threshold (overrides global if set).
    pub outlier_z_threshold: Option<f64>,
}

impl FieldConfig {
    /// Create a field config with the given name and normalisation.
    pub fn new(name: impl Into<String>, norm: NormStrategy) -> Self {
        Self {
            name: name.into(),
            norm,
            impute: ImputeStrategy::ForwardFill,
            outlier_detection: false,
            outlier_z_threshold: None,
        }
    }

    /// Create a field config with default settings.
    pub fn with_defaults(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            norm: NormStrategy::None,
            impute: ImputeStrategy::ForwardFill,
            outlier_detection: false,
            outlier_z_threshold: None,
        }
    }

    /// Enable outlier detection (builder pattern).
    pub fn with_outlier_detection(mut self, threshold: Option<f64>) -> Self {
        self.outlier_detection = true;
        self.outlier_z_threshold = threshold;
        self
    }

    /// Set imputation strategy (builder pattern).
    pub fn with_imputation(mut self, strategy: ImputeStrategy) -> Self {
        self.impute = strategy;
        self
    }
}

// ---------------------------------------------------------------------------
// Running statistics (Welford's algorithm)
// ---------------------------------------------------------------------------

/// Online running statistics for a single field.
#[derive(Debug, Clone)]
struct RunningStats {
    count: u64,
    mean: f64,
    m2: f64,
    min: f64,
    max: f64,
    last_value: Option<f64>,
}

impl RunningStats {
    fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            last_value: None,
        }
    }

    fn update(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
        if value < self.min {
            self.min = value;
        }
        if value > self.max {
            self.max = value;
        }
        self.last_value = Some(value);
    }

    fn variance(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            self.m2 / (self.count - 1) as f64
        }
    }

    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    fn reset(&mut self) {
        self.count = 0;
        self.mean = 0.0;
        self.m2 = 0.0;
        self.min = f64::INFINITY;
        self.max = f64::NEG_INFINITY;
        self.last_value = None;
    }
}

// ---------------------------------------------------------------------------
// Windowed buffer
// ---------------------------------------------------------------------------

/// Sliding window buffer for windowed statistics.
#[derive(Debug, Clone)]
struct WindowBuffer {
    buf: VecDeque<f64>,
    capacity: usize,
    sum: f64,
    sum_sq: f64,
}

impl WindowBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            buf: VecDeque::with_capacity(capacity.min(256)),
            capacity,
            sum: 0.0,
            sum_sq: 0.0,
        }
    }

    fn push(&mut self, value: f64) {
        if self.buf.len() >= self.capacity {
            if let Some(old) = self.buf.pop_front() {
                self.sum -= old;
                self.sum_sq -= old * old;
            }
        }
        self.buf.push_back(value);
        self.sum += value;
        self.sum_sq += value * value;
    }

    fn len(&self) -> usize {
        self.buf.len()
    }

    fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    fn mean(&self) -> f64 {
        if self.buf.is_empty() {
            0.0
        } else {
            self.sum / self.buf.len() as f64
        }
    }

    fn std_dev(&self) -> f64 {
        let n = self.buf.len();
        if n < 2 {
            return 0.0;
        }
        let mean = self.sum / n as f64;
        let variance = (self.sum_sq / n as f64) - (mean * mean);
        if variance <= 0.0 {
            0.0
        } else {
            variance.sqrt()
        }
    }

    fn median(&self) -> f64 {
        if self.buf.is_empty() {
            return 0.0;
        }
        let mut sorted: Vec<f64> = self.buf.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 0 {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        }
    }

    fn percentile(&self, p: f64) -> f64 {
        if self.buf.is_empty() {
            return 0.0;
        }
        let mut sorted: Vec<f64> = self.buf.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = (p / 100.0 * (sorted.len() - 1) as f64).round() as usize;
        sorted[idx.min(sorted.len() - 1)]
    }

    fn clear(&mut self) {
        self.buf.clear();
        self.sum = 0.0;
        self.sum_sq = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Per-field state
// ---------------------------------------------------------------------------

/// Internal state for a single registered field.
#[derive(Debug, Clone)]
struct FieldState {
    config: FieldConfig,
    running: RunningStats,
    window: WindowBuffer,
}

impl FieldState {
    fn new(config: FieldConfig, window_size: usize) -> Self {
        Self {
            config,
            running: RunningStats::new(),
            window: WindowBuffer::new(window_size),
        }
    }

    fn reset(&mut self) {
        self.running.reset();
        self.window.clear();
    }
}

// ---------------------------------------------------------------------------
// Processing result
// ---------------------------------------------------------------------------

/// Result of preprocessing a single value.
#[derive(Debug, Clone)]
pub struct ProcessedValue {
    /// The normalised / cleaned value.
    pub value: f64,
    /// The original raw value (before processing).
    pub raw: f64,
    /// Whether this value was identified as an outlier.
    pub is_outlier: bool,
    /// Whether this value was imputed (original was missing / NaN).
    pub was_imputed: bool,
    /// Whether the value was clamped.
    pub was_clamped: bool,
    /// The z-score of the raw value (if computed).
    pub z_score: Option<f64>,
}

/// Result of preprocessing a batch of field values.
#[derive(Debug, Clone)]
pub struct ProcessedBatch {
    /// Field name → processed value.
    pub values: HashMap<String, ProcessedValue>,
    /// Number of outliers detected.
    pub outlier_count: usize,
    /// Number of values imputed.
    pub imputed_count: usize,
}

// ---------------------------------------------------------------------------
// Public statistics
// ---------------------------------------------------------------------------

/// Public statistics for a single field.
#[derive(Debug, Clone)]
pub struct FieldStatsSummary {
    /// Field name.
    pub name: String,
    /// Number of observations.
    pub count: u64,
    /// Running mean.
    pub mean: f64,
    /// Running standard deviation.
    pub std_dev: f64,
    /// All-time minimum.
    pub min: f64,
    /// All-time maximum.
    pub max: f64,
    /// Windowed mean.
    pub windowed_mean: f64,
    /// Windowed standard deviation.
    pub windowed_std_dev: f64,
    /// Number of values in the sliding window.
    pub window_size: usize,
    /// Last observed value.
    pub last_value: Option<f64>,
}

/// Global preprocessing statistics.
#[derive(Debug, Clone)]
pub struct PreprocessingStats {
    /// Total values processed.
    pub total_processed: u64,
    /// Total outliers detected.
    pub total_outliers: u64,
    /// Total values imputed.
    pub total_imputed: u64,
    /// Total non-finite values filtered.
    pub total_non_finite: u64,
    /// Total values clamped.
    pub total_clamped: u64,
    /// Number of registered fields.
    pub field_count: usize,
    /// Number of ticks processed.
    pub ticks: u64,
}

impl Default for PreprocessingStats {
    fn default() -> Self {
        Self {
            total_processed: 0,
            total_outliers: 0,
            total_imputed: 0,
            total_non_finite: 0,
            total_clamped: 0,
            field_count: 0,
            ticks: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Preprocessing
// ---------------------------------------------------------------------------

/// Data normalization and cleaning engine.
///
/// Multi-field preprocessing pipeline with outlier detection, imputation,
/// normalisation, and running/windowed statistics.
pub struct Preprocessing {
    config: PreprocessingConfig,
    fields: HashMap<String, FieldState>,
    field_order: Vec<String>,

    // Counters
    total_processed: u64,
    total_outliers: u64,
    total_imputed: u64,
    total_non_finite: u64,
    total_clamped: u64,
    current_tick: u64,
}

impl Default for Preprocessing {
    fn default() -> Self {
        Self::new()
    }
}

impl Preprocessing {
    /// Create a new preprocessing engine with default configuration.
    pub fn new() -> Self {
        Self::with_config(PreprocessingConfig::default()).unwrap()
    }

    /// Create with explicit configuration.
    pub fn with_config(config: PreprocessingConfig) -> Result<Self> {
        if config.max_fields == 0 {
            return Err(Error::InvalidInput("max_fields must be > 0".into()));
        }
        if config.window_size == 0 {
            return Err(Error::InvalidInput("window_size must be > 0".into()));
        }
        if config.outlier_z_threshold <= 0.0 {
            return Err(Error::InvalidInput(
                "outlier_z_threshold must be > 0".into(),
            ));
        }
        Ok(Self {
            config,
            fields: HashMap::new(),
            field_order: Vec::new(),
            total_processed: 0,
            total_outliers: 0,
            total_imputed: 0,
            total_non_finite: 0,
            total_clamped: 0,
            current_tick: 0,
        })
    }

    // -----------------------------------------------------------------------
    // Field management
    // -----------------------------------------------------------------------

    /// Register a field with a specific configuration.
    pub fn register_field(&mut self, config: FieldConfig) -> Result<()> {
        if self.fields.contains_key(&config.name) {
            return Err(Error::InvalidInput(format!(
                "field '{}' already registered",
                config.name
            )));
        }
        if self.fields.len() >= self.config.max_fields {
            return Err(Error::ResourceExhausted(format!(
                "maximum fields ({}) reached",
                self.config.max_fields
            )));
        }
        let name = config.name.clone();
        self.fields.insert(
            name.clone(),
            FieldState::new(config, self.config.window_size),
        );
        self.field_order.push(name);
        Ok(())
    }

    /// Register a field with a name and normalisation strategy.
    pub fn register(&mut self, name: impl Into<String>, norm: NormStrategy) -> Result<()> {
        self.register_field(FieldConfig::new(name, norm))
    }

    /// Register a field with default settings.
    pub fn register_default(&mut self, name: impl Into<String>) -> Result<()> {
        let n = name.into();
        self.register_field(FieldConfig::new(n, self.config.default_strategy))
    }

    /// Deregister a field.
    pub fn deregister(&mut self, name: &str) -> Result<()> {
        if self.fields.remove(name).is_none() {
            return Err(Error::NotFound(format!("field '{}' not found", name)));
        }
        self.field_order.retain(|n| n != name);
        Ok(())
    }

    /// Get the list of registered field names (in registration order).
    pub fn field_names(&self) -> &[String] {
        &self.field_order
    }

    /// Number of registered fields.
    pub fn field_count(&self) -> usize {
        self.fields.len()
    }

    // -----------------------------------------------------------------------
    // Process single value
    // -----------------------------------------------------------------------

    /// Process a single value for a named field.
    ///
    /// Applies (in order): non-finite filtering → outlier detection →
    /// imputation → normalisation.
    pub fn process_value(&mut self, field_name: &str, value: f64) -> Result<ProcessedValue> {
        let state = self
            .fields
            .get_mut(field_name)
            .ok_or_else(|| Error::NotFound(format!("field '{}' not found", field_name)))?;

        let raw = value;
        let mut current = value;
        let mut is_outlier = false;
        let mut was_imputed = false;
        let mut was_clamped = false;
        let mut z_score: Option<f64> = None;

        // Step 1: Non-finite filtering
        if !current.is_finite() && self.config.filter_non_finite {
            self.total_non_finite += 1;
            // Treat as missing → impute
            current = self.impute_value(state);
            was_imputed = true;
            self.total_imputed += 1;
        }

        // Step 2: Outlier detection (on raw value, only if we have enough data)
        if current.is_finite()
            && state.config.outlier_detection
            && state.running.count >= self.config.min_observations as u64
        {
            let w_mean = state.window.mean();
            let w_std = state.window.std_dev();
            if w_std > 1e-15 {
                let z = (current - w_mean) / w_std;
                z_score = Some(z);
                let threshold = state
                    .config
                    .outlier_z_threshold
                    .unwrap_or(self.config.outlier_z_threshold);
                if z.abs() > threshold {
                    is_outlier = true;
                    self.total_outliers += 1;
                    if self.config.clamp_outliers {
                        // Clamp to threshold boundary
                        current = w_mean + z.signum() * threshold * w_std;
                        was_clamped = true;
                        self.total_clamped += 1;
                    }
                }
            }
        }

        // Step 3: Update running statistics (with potentially clamped value)
        if current.is_finite() {
            state.running.update(current);
            state.window.push(current);
        }

        // Step 4: Normalise
        let normalised = if state.running.count >= self.config.min_observations as u64 {
            self.normalise(current, state)
        } else {
            current
        };

        self.total_processed += 1;

        Ok(ProcessedValue {
            value: normalised,
            raw,
            is_outlier,
            was_imputed,
            was_clamped,
            z_score,
        })
    }

    /// Process a missing/null value for a named field.
    ///
    /// Uses the configured imputation strategy.
    pub fn process_missing(&mut self, field_name: &str) -> Result<ProcessedValue> {
        let state = self
            .fields
            .get_mut(field_name)
            .ok_or_else(|| Error::NotFound(format!("field '{}' not found", field_name)))?;

        let imputed = self.impute_value(state);
        self.total_imputed += 1;

        // Update stats with imputed value
        if imputed.is_finite() {
            state.running.update(imputed);
            state.window.push(imputed);
        }

        let normalised = if state.running.count >= self.config.min_observations as u64 {
            self.normalise(imputed, state)
        } else {
            imputed
        };

        self.total_processed += 1;

        Ok(ProcessedValue {
            value: normalised,
            raw: f64::NAN,
            is_outlier: false,
            was_imputed: true,
            was_clamped: false,
            z_score: None,
        })
    }

    /// Process a batch of (field_name, value) pairs.
    pub fn process_batch(&mut self, entries: &[(&str, f64)]) -> Result<ProcessedBatch> {
        let mut values = HashMap::new();
        let mut outlier_count = 0;
        let mut imputed_count = 0;

        for &(field_name, value) in entries {
            let result = self.process_value(field_name, value)?;
            if result.is_outlier {
                outlier_count += 1;
            }
            if result.was_imputed {
                imputed_count += 1;
            }
            values.insert(field_name.to_string(), result);
        }

        Ok(ProcessedBatch {
            values,
            outlier_count,
            imputed_count,
        })
    }

    /// Process a series of values for a single field.
    ///
    /// Returns a vector of processed values.
    pub fn process_series(
        &mut self,
        field_name: &str,
        values: &[f64],
    ) -> Result<Vec<ProcessedValue>> {
        let mut results = Vec::with_capacity(values.len());
        for &v in values {
            results.push(self.process_value(field_name, v)?);
        }
        Ok(results)
    }

    // -----------------------------------------------------------------------
    // Normalisation internals
    // -----------------------------------------------------------------------

    fn normalise(&self, value: f64, state: &FieldState) -> f64 {
        match state.config.norm {
            NormStrategy::None => value,
            NormStrategy::MinMax => {
                let range = state.running.max - state.running.min;
                if range.abs() < 1e-15 {
                    0.5
                } else {
                    ((value - state.running.min) / range).clamp(0.0, 1.0)
                }
            }
            NormStrategy::ZScore => {
                let sd = state.running.std_dev();
                if sd.abs() < 1e-15 {
                    0.0
                } else {
                    (value - state.running.mean) / sd
                }
            }
            NormStrategy::Robust => {
                if state.window.len() < 4 {
                    // Fall back to z-score
                    let sd = state.running.std_dev();
                    if sd.abs() < 1e-15 {
                        0.0
                    } else {
                        (value - state.running.mean) / sd
                    }
                } else {
                    let median = state.window.median();
                    let q1 = state.window.percentile(25.0);
                    let q3 = state.window.percentile(75.0);
                    let iqr = q3 - q1;
                    if iqr.abs() < 1e-15 {
                        0.0
                    } else {
                        (value - median) / iqr
                    }
                }
            }
            NormStrategy::Log => {
                if value <= 0.0 {
                    f64::NEG_INFINITY
                } else {
                    value.ln()
                }
            }
            NormStrategy::PctChange => {
                if let Some(prev) = state.running.last_value {
                    if prev.abs() < 1e-15 {
                        0.0
                    } else {
                        (value - prev) / prev.abs()
                    }
                } else {
                    0.0
                }
            }
        }
    }

    fn impute_value(&self, state: &FieldState) -> f64 {
        let strategy = state.config.impute;
        match strategy {
            ImputeStrategy::ForwardFill => state.running.last_value.unwrap_or(0.0),
            ImputeStrategy::Mean => {
                if state.running.count > 0 {
                    state.running.mean
                } else {
                    0.0
                }
            }
            ImputeStrategy::Median => {
                if state.window.is_empty() {
                    0.0
                } else {
                    state.window.median()
                }
            }
            ImputeStrategy::Zero => 0.0,
            ImputeStrategy::Skip => f64::NAN,
        }
    }

    // -----------------------------------------------------------------------
    // Queries
    // -----------------------------------------------------------------------

    /// Get statistics for a specific field.
    pub fn field_stats(&self, name: &str) -> Result<FieldStatsSummary> {
        let state = self
            .fields
            .get(name)
            .ok_or_else(|| Error::NotFound(format!("field '{}' not found", name)))?;
        Ok(FieldStatsSummary {
            name: name.to_string(),
            count: state.running.count,
            mean: state.running.mean,
            std_dev: state.running.std_dev(),
            min: state.running.min,
            max: state.running.max,
            windowed_mean: state.window.mean(),
            windowed_std_dev: state.window.std_dev(),
            window_size: state.window.len(),
            last_value: state.running.last_value,
        })
    }

    /// Compute the current z-score for a value in a field without processing it.
    pub fn z_score(&self, field_name: &str, value: f64) -> Result<Option<f64>> {
        let state = self
            .fields
            .get(field_name)
            .ok_or_else(|| Error::NotFound(format!("field '{}' not found", field_name)))?;
        if state.window.len() < 2 {
            return Ok(None);
        }
        let w_mean = state.window.mean();
        let w_std = state.window.std_dev();
        if w_std.abs() < 1e-15 {
            return Ok(Some(0.0));
        }
        Ok(Some((value - w_mean) / w_std))
    }

    /// Get the windowed mean for a field.
    pub fn windowed_mean(&self, field_name: &str) -> Result<f64> {
        let state = self
            .fields
            .get(field_name)
            .ok_or_else(|| Error::NotFound(format!("field '{}' not found", field_name)))?;
        Ok(state.window.mean())
    }

    /// Get the windowed standard deviation for a field.
    pub fn windowed_std_dev(&self, field_name: &str) -> Result<f64> {
        let state = self
            .fields
            .get(field_name)
            .ok_or_else(|| Error::NotFound(format!("field '{}' not found", field_name)))?;
        Ok(state.window.std_dev())
    }

    /// Whether a field has enough observations for normalisation.
    pub fn is_warmed_up(&self, field_name: &str) -> Result<bool> {
        let state = self
            .fields
            .get(field_name)
            .ok_or_else(|| Error::NotFound(format!("field '{}' not found", field_name)))?;
        Ok(state.running.count >= self.config.min_observations as u64)
    }

    /// Current tick.
    pub fn current_tick(&self) -> u64 {
        self.current_tick
    }

    // -----------------------------------------------------------------------
    // Tick / Process / Stats
    // -----------------------------------------------------------------------

    /// Advance one tick.
    pub fn tick(&mut self) {
        self.current_tick += 1;
    }

    /// Global preprocessing statistics.
    pub fn stats(&self) -> PreprocessingStats {
        PreprocessingStats {
            total_processed: self.total_processed,
            total_outliers: self.total_outliers,
            total_imputed: self.total_imputed,
            total_non_finite: self.total_non_finite,
            total_clamped: self.total_clamped,
            field_count: self.fields.len(),
            ticks: self.current_tick,
        }
    }

    // -----------------------------------------------------------------------
    // Clear / Reset
    // -----------------------------------------------------------------------

    /// Reset statistics for a single field.
    pub fn reset_field(&mut self, name: &str) -> Result<()> {
        let state = self
            .fields
            .get_mut(name)
            .ok_or_else(|| Error::NotFound(format!("field '{}' not found", name)))?;
        state.reset();
        Ok(())
    }

    /// Reset all field statistics but keep registrations.
    pub fn reset_stats(&mut self) {
        for state in self.fields.values_mut() {
            state.reset();
        }
        self.total_processed = 0;
        self.total_outliers = 0;
        self.total_imputed = 0;
        self.total_non_finite = 0;
        self.total_clamped = 0;
    }

    /// Full reset — clears everything including field registrations.
    pub fn reset(&mut self) {
        self.fields.clear();
        self.field_order.clear();
        self.total_processed = 0;
        self.total_outliers = 0;
        self.total_imputed = 0;
        self.total_non_finite = 0;
        self.total_clamped = 0;
        self.current_tick = 0;
    }

    /// Main processing function (tick alias).
    pub fn process(&mut self) -> Result<()> {
        self.tick();
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
    // Construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_basic() {
        let pp = Preprocessing::new();
        assert_eq!(pp.field_count(), 0);
        assert_eq!(pp.current_tick(), 0);
    }

    #[test]
    fn test_default() {
        let pp = Preprocessing::default();
        assert_eq!(pp.field_count(), 0);
    }

    #[test]
    fn test_with_config() {
        let cfg = PreprocessingConfig {
            max_fields: 8,
            window_size: 50,
            outlier_z_threshold: 2.0,
            ..Default::default()
        };
        let pp = Preprocessing::with_config(cfg).unwrap();
        assert_eq!(pp.field_count(), 0);
    }

    #[test]
    fn test_invalid_config_max_fields() {
        let mut cfg = PreprocessingConfig::default();
        cfg.max_fields = 0;
        assert!(Preprocessing::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_window_size() {
        let mut cfg = PreprocessingConfig::default();
        cfg.window_size = 0;
        assert!(Preprocessing::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_z_threshold() {
        let mut cfg = PreprocessingConfig::default();
        cfg.outlier_z_threshold = 0.0;
        assert!(Preprocessing::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_z_threshold_negative() {
        let mut cfg = PreprocessingConfig::default();
        cfg.outlier_z_threshold = -1.0;
        assert!(Preprocessing::with_config(cfg).is_err());
    }

    // -----------------------------------------------------------------------
    // Field management
    // -----------------------------------------------------------------------

    #[test]
    fn test_register_field() {
        let mut pp = Preprocessing::new();
        pp.register("price", NormStrategy::MinMax).unwrap();
        assert_eq!(pp.field_count(), 1);
        assert_eq!(pp.field_names(), &["price"]);
    }

    #[test]
    fn test_register_field_config() {
        let mut pp = Preprocessing::new();
        let cfg = FieldConfig::new("price", NormStrategy::ZScore)
            .with_outlier_detection(Some(2.5))
            .with_imputation(ImputeStrategy::Mean);
        pp.register_field(cfg).unwrap();
        assert_eq!(pp.field_count(), 1);
    }

    #[test]
    fn test_register_default() {
        let mut pp = Preprocessing::new();
        pp.register_default("price").unwrap();
        assert_eq!(pp.field_count(), 1);
    }

    #[test]
    fn test_register_duplicate() {
        let mut pp = Preprocessing::new();
        pp.register("price", NormStrategy::None).unwrap();
        assert!(pp.register("price", NormStrategy::None).is_err());
    }

    #[test]
    fn test_register_max_capacity() {
        let cfg = PreprocessingConfig {
            max_fields: 2,
            ..Default::default()
        };
        let mut pp = Preprocessing::with_config(cfg).unwrap();
        pp.register("a", NormStrategy::None).unwrap();
        pp.register("b", NormStrategy::None).unwrap();
        assert!(pp.register("c", NormStrategy::None).is_err());
    }

    #[test]
    fn test_deregister() {
        let mut pp = Preprocessing::new();
        pp.register("price", NormStrategy::None).unwrap();
        pp.deregister("price").unwrap();
        assert_eq!(pp.field_count(), 0);
    }

    #[test]
    fn test_deregister_nonexistent() {
        let mut pp = Preprocessing::new();
        assert!(pp.deregister("nope").is_err());
    }

    #[test]
    fn test_field_order_preserved() {
        let mut pp = Preprocessing::new();
        pp.register("c", NormStrategy::None).unwrap();
        pp.register("a", NormStrategy::None).unwrap();
        pp.register("b", NormStrategy::None).unwrap();
        assert_eq!(pp.field_names(), &["c", "a", "b"]);
    }

    #[test]
    fn test_deregister_preserves_order() {
        let mut pp = Preprocessing::new();
        pp.register("a", NormStrategy::None).unwrap();
        pp.register("b", NormStrategy::None).unwrap();
        pp.register("c", NormStrategy::None).unwrap();
        pp.deregister("b").unwrap();
        assert_eq!(pp.field_names(), &["a", "c"]);
    }

    // -----------------------------------------------------------------------
    // No normalisation (pass-through)
    // -----------------------------------------------------------------------

    #[test]
    fn test_process_value_none() {
        let mut pp = Preprocessing::new();
        pp.register("price", NormStrategy::None).unwrap();
        let result = pp.process_value("price", 42.0).unwrap();
        assert!((result.value - 42.0).abs() < 1e-10);
        assert!((result.raw - 42.0).abs() < 1e-10);
        assert!(!result.is_outlier);
        assert!(!result.was_imputed);
    }

    #[test]
    fn test_process_value_nonexistent() {
        let mut pp = Preprocessing::new();
        assert!(pp.process_value("nope", 1.0).is_err());
    }

    // -----------------------------------------------------------------------
    // Min-max normalisation
    // -----------------------------------------------------------------------

    #[test]
    fn test_minmax_normalisation() {
        let cfg = PreprocessingConfig {
            min_observations: 3,
            ..Default::default()
        };
        let mut pp = Preprocessing::with_config(cfg).unwrap();
        pp.register("price", NormStrategy::MinMax).unwrap();

        // Push 3 values to warm up: 10, 20, 30
        pp.process_value("price", 10.0).unwrap();
        pp.process_value("price", 20.0).unwrap();
        pp.process_value("price", 30.0).unwrap();

        // Now normalisation should be active
        // min=10, max=30, range=20
        // Value 20 → (20-10)/20 = 0.5
        let result = pp.process_value("price", 20.0).unwrap();
        assert!((result.value - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_minmax_constant_values() {
        let cfg = PreprocessingConfig {
            min_observations: 2,
            ..Default::default()
        };
        let mut pp = Preprocessing::with_config(cfg).unwrap();
        pp.register("price", NormStrategy::MinMax).unwrap();

        pp.process_value("price", 5.0).unwrap();
        pp.process_value("price", 5.0).unwrap();

        // Constant values → 0.5
        let result = pp.process_value("price", 5.0).unwrap();
        assert!((result.value - 0.5).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Z-score normalisation
    // -----------------------------------------------------------------------

    #[test]
    fn test_zscore_normalisation() {
        let cfg = PreprocessingConfig {
            min_observations: 5,
            ..Default::default()
        };
        let mut pp = Preprocessing::with_config(cfg).unwrap();
        pp.register("price", NormStrategy::ZScore).unwrap();

        // Push 5 identical values → mean = 100, std = 0
        for _ in 0..5 {
            pp.process_value("price", 100.0).unwrap();
        }

        // Constant values → z-score = 0
        let result = pp.process_value("price", 100.0).unwrap();
        assert!((result.value - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_zscore_varying() {
        let cfg = PreprocessingConfig {
            min_observations: 3,
            ..Default::default()
        };
        let mut pp = Preprocessing::with_config(cfg).unwrap();
        pp.register("price", NormStrategy::ZScore).unwrap();

        for i in 0..100 {
            pp.process_value("price", i as f64).unwrap();
        }

        let stats = pp.field_stats("price").unwrap();
        assert!(stats.std_dev > 0.0);
        assert!(stats.count == 100);
    }

    // -----------------------------------------------------------------------
    // Log normalisation
    // -----------------------------------------------------------------------

    #[test]
    fn test_log_normalisation() {
        let cfg = PreprocessingConfig {
            min_observations: 1,
            ..Default::default()
        };
        let mut pp = Preprocessing::with_config(cfg).unwrap();
        pp.register("price", NormStrategy::Log).unwrap();

        pp.process_value("price", 1.0).unwrap();
        let result = pp.process_value("price", std::f64::consts::E).unwrap();
        assert!((result.value - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_log_negative_value() {
        let cfg = PreprocessingConfig {
            min_observations: 1,
            filter_non_finite: false,
            ..Default::default()
        };
        let mut pp = Preprocessing::with_config(cfg).unwrap();
        pp.register("price", NormStrategy::Log).unwrap();

        pp.process_value("price", 1.0).unwrap();
        let result = pp.process_value("price", -1.0).unwrap();
        assert!(result.value.is_infinite());
    }

    // -----------------------------------------------------------------------
    // Percentage change
    // -----------------------------------------------------------------------

    #[test]
    fn test_pct_change() {
        let cfg = PreprocessingConfig {
            min_observations: 1,
            ..Default::default()
        };
        let mut pp = Preprocessing::with_config(cfg).unwrap();
        pp.register("price", NormStrategy::PctChange).unwrap();

        // First value has no predecessor → 0
        let r1 = pp.process_value("price", 100.0).unwrap();
        assert!((r1.value - 0.0).abs() < 1e-10);

        // Second value: pct change = (110-100)/100 = 0.1
        let r2 = pp.process_value("price", 110.0).unwrap();
        assert!((r2.value - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_pct_change_from_zero() {
        let cfg = PreprocessingConfig {
            min_observations: 1,
            ..Default::default()
        };
        let mut pp = Preprocessing::with_config(cfg).unwrap();
        pp.register("price", NormStrategy::PctChange).unwrap();

        pp.process_value("price", 0.0).unwrap();
        // pct change from zero → 0 (avoid division by zero)
        let result = pp.process_value("price", 10.0).unwrap();
        assert!((result.value - 0.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Robust normalisation
    // -----------------------------------------------------------------------

    #[test]
    fn test_robust_normalisation() {
        let cfg = PreprocessingConfig {
            min_observations: 5,
            window_size: 20,
            ..Default::default()
        };
        let mut pp = Preprocessing::with_config(cfg).unwrap();
        pp.register("price", NormStrategy::Robust).unwrap();

        for i in 0..20 {
            pp.process_value("price", i as f64).unwrap();
        }

        // After 20 observations, robust scaling should use median / IQR
        let result = pp.process_value("price", 10.0).unwrap();
        // Just check it produces a finite value
        assert!(result.value.is_finite());
    }

    #[test]
    fn test_robust_falls_back_to_zscore() {
        let cfg = PreprocessingConfig {
            min_observations: 2,
            window_size: 100,
            ..Default::default()
        };
        let mut pp = Preprocessing::with_config(cfg).unwrap();
        pp.register("price", NormStrategy::Robust).unwrap();

        // Only 3 values → window has < 4 → falls back to z-score
        pp.process_value("price", 10.0).unwrap();
        pp.process_value("price", 20.0).unwrap();
        let result = pp.process_value("price", 15.0).unwrap();
        assert!(result.value.is_finite());
    }

    // -----------------------------------------------------------------------
    // Outlier detection
    // -----------------------------------------------------------------------

    #[test]
    fn test_outlier_detection_clamp() {
        let cfg = PreprocessingConfig {
            min_observations: 5,
            window_size: 20,
            outlier_z_threshold: 2.0,
            clamp_outliers: true,
            ..Default::default()
        };
        let mut pp = Preprocessing::with_config(cfg).unwrap();
        let field_cfg = FieldConfig::new("price", NormStrategy::None).with_outlier_detection(None);
        pp.register_field(field_cfg).unwrap();

        // Push 20 values in tight range
        for i in 0..20 {
            pp.process_value("price", 100.0 + (i % 3) as f64).unwrap();
        }

        // Now push a huge outlier
        let result = pp.process_value("price", 1_000_000.0).unwrap();
        assert!(result.is_outlier);
        assert!(result.was_clamped);
        // The clamped value should be much less than 1M
        assert!(result.value < 1_000_000.0);
    }

    #[test]
    fn test_outlier_detection_no_clamp() {
        let cfg = PreprocessingConfig {
            min_observations: 5,
            window_size: 20,
            outlier_z_threshold: 2.0,
            clamp_outliers: false,
            ..Default::default()
        };
        let mut pp = Preprocessing::with_config(cfg).unwrap();
        let field_cfg = FieldConfig::new("price", NormStrategy::None).with_outlier_detection(None);
        pp.register_field(field_cfg).unwrap();

        for i in 0..20 {
            pp.process_value("price", 100.0 + (i % 3) as f64).unwrap();
        }

        let result = pp.process_value("price", 1_000_000.0).unwrap();
        assert!(result.is_outlier);
        assert!(!result.was_clamped);
    }

    #[test]
    fn test_outlier_not_triggered_below_threshold() {
        let cfg = PreprocessingConfig {
            min_observations: 5,
            window_size: 20,
            outlier_z_threshold: 3.0,
            ..Default::default()
        };
        let mut pp = Preprocessing::with_config(cfg).unwrap();
        let field_cfg = FieldConfig::new("price", NormStrategy::None).with_outlier_detection(None);
        pp.register_field(field_cfg).unwrap();

        for i in 0..20 {
            pp.process_value("price", 100.0 + (i % 3) as f64).unwrap();
        }

        // Push a value that's within normal range
        let result = pp.process_value("price", 101.0).unwrap();
        assert!(!result.is_outlier);
    }

    #[test]
    fn test_outlier_not_active_before_warmup() {
        let cfg = PreprocessingConfig {
            min_observations: 100,
            ..Default::default()
        };
        let mut pp = Preprocessing::with_config(cfg).unwrap();
        let field_cfg = FieldConfig::new("price", NormStrategy::None).with_outlier_detection(None);
        pp.register_field(field_cfg).unwrap();

        // Only 5 values → not warmed up → outlier detection inactive
        for _ in 0..5 {
            pp.process_value("price", 100.0).unwrap();
        }
        let result = pp.process_value("price", 1_000_000.0).unwrap();
        assert!(!result.is_outlier);
    }

    #[test]
    fn test_outlier_custom_threshold() {
        let cfg = PreprocessingConfig {
            min_observations: 5,
            window_size: 20,
            outlier_z_threshold: 10.0, // global is very high
            clamp_outliers: true,
            ..Default::default()
        };
        let mut pp = Preprocessing::with_config(cfg).unwrap();
        // Field has a low custom threshold
        let field_cfg =
            FieldConfig::new("price", NormStrategy::None).with_outlier_detection(Some(1.5));
        pp.register_field(field_cfg).unwrap();

        for _ in 0..20 {
            pp.process_value("price", 100.0).unwrap();
        }

        // Even a moderate outlier should be caught with threshold 1.5
        let result = pp.process_value("price", 200.0).unwrap();
        // The value is far from 100, so it should trigger with threshold 1.5
        // (depends on std dev, but 100 → 200 with std ≈ 0 should trigger)
        // Actually all values are 100.0, so std ≈ 0, and z_score won't be computed
        // since w_std < 1e-15. Let's use varying data:
        // Re-test with varying data
        let mut pp2 = Preprocessing::with_config(PreprocessingConfig {
            min_observations: 5,
            window_size: 20,
            outlier_z_threshold: 10.0,
            clamp_outliers: true,
            ..Default::default()
        })
        .unwrap();
        let field_cfg2 =
            FieldConfig::new("x", NormStrategy::None).with_outlier_detection(Some(1.5));
        pp2.register_field(field_cfg2).unwrap();

        for i in 0..20 {
            pp2.process_value("x", 100.0 + (i as f64 * 0.5)).unwrap();
        }
        let result2 = pp2.process_value("x", 500.0).unwrap();
        assert!(result2.is_outlier);
        assert!(result2.z_score.unwrap() > 1.5);
    }

    // -----------------------------------------------------------------------
    // Missing value imputation
    // -----------------------------------------------------------------------

    #[test]
    fn test_impute_forward_fill() {
        let mut pp = Preprocessing::new();
        let cfg = FieldConfig::new("price", NormStrategy::None)
            .with_imputation(ImputeStrategy::ForwardFill);
        pp.register_field(cfg).unwrap();

        pp.process_value("price", 42.0).unwrap();
        let result = pp.process_missing("price").unwrap();
        assert!(result.was_imputed);
        assert!((result.value - 42.0).abs() < 1e-10);
    }

    #[test]
    fn test_impute_forward_fill_no_previous() {
        let mut pp = Preprocessing::new();
        let cfg = FieldConfig::new("price", NormStrategy::None)
            .with_imputation(ImputeStrategy::ForwardFill);
        pp.register_field(cfg).unwrap();

        // No previous value → falls back to 0
        let result = pp.process_missing("price").unwrap();
        assert!(result.was_imputed);
        assert!((result.value - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_impute_mean() {
        let mut pp = Preprocessing::new();
        let cfg =
            FieldConfig::new("price", NormStrategy::None).with_imputation(ImputeStrategy::Mean);
        pp.register_field(cfg).unwrap();

        pp.process_value("price", 10.0).unwrap();
        pp.process_value("price", 20.0).unwrap();
        pp.process_value("price", 30.0).unwrap();

        let result = pp.process_missing("price").unwrap();
        assert!(result.was_imputed);
        assert!((result.value - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_impute_zero() {
        let mut pp = Preprocessing::new();
        let cfg =
            FieldConfig::new("price", NormStrategy::None).with_imputation(ImputeStrategy::Zero);
        pp.register_field(cfg).unwrap();

        pp.process_value("price", 42.0).unwrap();
        let result = pp.process_missing("price").unwrap();
        assert!((result.value - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_impute_median() {
        let mut pp = Preprocessing::new();
        let cfg =
            FieldConfig::new("price", NormStrategy::None).with_imputation(ImputeStrategy::Median);
        pp.register_field(cfg).unwrap();

        pp.process_value("price", 10.0).unwrap();
        pp.process_value("price", 20.0).unwrap();
        pp.process_value("price", 30.0).unwrap();

        let result = pp.process_missing("price").unwrap();
        assert!(result.was_imputed);
        assert!((result.value - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_impute_skip() {
        let mut pp = Preprocessing::new();
        let cfg =
            FieldConfig::new("price", NormStrategy::None).with_imputation(ImputeStrategy::Skip);
        pp.register_field(cfg).unwrap();

        let result = pp.process_missing("price").unwrap();
        assert!(result.value.is_nan());
    }

    #[test]
    fn test_process_missing_nonexistent() {
        let mut pp = Preprocessing::new();
        assert!(pp.process_missing("nope").is_err());
    }

    // -----------------------------------------------------------------------
    // Non-finite value filtering
    // -----------------------------------------------------------------------

    #[test]
    fn test_nan_filtered() {
        let mut pp = Preprocessing::new();
        let cfg =
            FieldConfig::new("price", NormStrategy::None).with_imputation(ImputeStrategy::Zero);
        pp.register_field(cfg).unwrap();

        let result = pp.process_value("price", f64::NAN).unwrap();
        assert!(result.was_imputed);
        assert!((result.value - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_inf_filtered() {
        let mut pp = Preprocessing::new();
        let cfg =
            FieldConfig::new("price", NormStrategy::None).with_imputation(ImputeStrategy::Zero);
        pp.register_field(cfg).unwrap();

        let result = pp.process_value("price", f64::INFINITY).unwrap();
        assert!(result.was_imputed);
        assert!((result.value - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_nan_not_filtered_when_disabled() {
        let cfg = PreprocessingConfig {
            filter_non_finite: false,
            ..Default::default()
        };
        let mut pp = Preprocessing::with_config(cfg).unwrap();
        pp.register("price", NormStrategy::None).unwrap();

        let result = pp.process_value("price", f64::NAN).unwrap();
        assert!(!result.was_imputed);
        assert!(result.value.is_nan());
    }

    // -----------------------------------------------------------------------
    // Series processing
    // -----------------------------------------------------------------------

    #[test]
    fn test_process_series() {
        let mut pp = Preprocessing::new();
        pp.register("price", NormStrategy::None).unwrap();

        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let results = pp.process_series("price", &values).unwrap();
        assert_eq!(results.len(), 5);
        for (i, r) in results.iter().enumerate() {
            assert!((r.value - (i + 1) as f64).abs() < 1e-10);
        }
    }

    #[test]
    fn test_process_series_nonexistent() {
        let mut pp = Preprocessing::new();
        assert!(pp.process_series("nope", &[1.0]).is_err());
    }

    // -----------------------------------------------------------------------
    // Batch processing
    // -----------------------------------------------------------------------

    #[test]
    fn test_process_batch() {
        let mut pp = Preprocessing::new();
        pp.register("price", NormStrategy::None).unwrap();
        pp.register("volume", NormStrategy::None).unwrap();

        let batch = pp
            .process_batch(&[("price", 100.0), ("volume", 500.0)])
            .unwrap();
        assert_eq!(batch.values.len(), 2);
        assert!((batch.values["price"].value - 100.0).abs() < 1e-10);
        assert!((batch.values["volume"].value - 500.0).abs() < 1e-10);
    }

    #[test]
    fn test_process_batch_nonexistent() {
        let mut pp = Preprocessing::new();
        assert!(pp.process_batch(&[("nope", 1.0)]).is_err());
    }

    // -----------------------------------------------------------------------
    // Field stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_field_stats() {
        let mut pp = Preprocessing::new();
        pp.register("price", NormStrategy::None).unwrap();

        pp.process_value("price", 10.0).unwrap();
        pp.process_value("price", 20.0).unwrap();
        pp.process_value("price", 30.0).unwrap();

        let stats = pp.field_stats("price").unwrap();
        assert_eq!(stats.count, 3);
        assert!((stats.mean - 20.0).abs() < 1e-10);
        assert!((stats.min - 10.0).abs() < 1e-10);
        assert!((stats.max - 30.0).abs() < 1e-10);
        assert!(stats.std_dev > 0.0);
        assert_eq!(stats.window_size, 3);
        assert!(stats.last_value.is_some());
        assert!((stats.last_value.unwrap() - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_field_stats_nonexistent() {
        let pp = Preprocessing::new();
        assert!(pp.field_stats("nope").is_err());
    }

    // -----------------------------------------------------------------------
    // z-score query
    // -----------------------------------------------------------------------

    #[test]
    fn test_z_score_query() {
        let mut pp = Preprocessing::new();
        pp.register("price", NormStrategy::None).unwrap();

        for i in 0..20 {
            pp.process_value("price", 100.0 + (i as f64 * 0.5)).unwrap();
        }

        let z = pp.z_score("price", 100.0).unwrap();
        assert!(z.is_some());
        // 100.0 is below the window mean, so z-score should be negative
        assert!(z.unwrap() < 0.0);
    }

    #[test]
    fn test_z_score_insufficient_data() {
        let mut pp = Preprocessing::new();
        pp.register("price", NormStrategy::None).unwrap();

        pp.process_value("price", 100.0).unwrap();
        let z = pp.z_score("price", 100.0).unwrap();
        assert!(z.is_none()); // only 1 value
    }

    #[test]
    fn test_z_score_nonexistent() {
        let pp = Preprocessing::new();
        assert!(pp.z_score("nope", 1.0).is_err());
    }

    // -----------------------------------------------------------------------
    // Windowed stats queries
    // -----------------------------------------------------------------------

    #[test]
    fn test_windowed_mean() {
        let mut pp = Preprocessing::new();
        pp.register("price", NormStrategy::None).unwrap();

        pp.process_value("price", 10.0).unwrap();
        pp.process_value("price", 20.0).unwrap();
        pp.process_value("price", 30.0).unwrap();

        let wm = pp.windowed_mean("price").unwrap();
        assert!((wm - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_windowed_std_dev() {
        let mut pp = Preprocessing::new();
        pp.register("price", NormStrategy::None).unwrap();

        pp.process_value("price", 10.0).unwrap();
        pp.process_value("price", 20.0).unwrap();
        pp.process_value("price", 30.0).unwrap();

        let ws = pp.windowed_std_dev("price").unwrap();
        assert!(ws > 0.0);
    }

    #[test]
    fn test_windowed_mean_nonexistent() {
        let pp = Preprocessing::new();
        assert!(pp.windowed_mean("nope").is_err());
    }

    #[test]
    fn test_windowed_std_dev_nonexistent() {
        let pp = Preprocessing::new();
        assert!(pp.windowed_std_dev("nope").is_err());
    }

    // -----------------------------------------------------------------------
    // is_warmed_up
    // -----------------------------------------------------------------------

    #[test]
    fn test_is_warmed_up() {
        let cfg = PreprocessingConfig {
            min_observations: 3,
            ..Default::default()
        };
        let mut pp = Preprocessing::with_config(cfg).unwrap();
        pp.register("price", NormStrategy::None).unwrap();

        assert!(!pp.is_warmed_up("price").unwrap());
        pp.process_value("price", 1.0).unwrap();
        pp.process_value("price", 2.0).unwrap();
        assert!(!pp.is_warmed_up("price").unwrap());
        pp.process_value("price", 3.0).unwrap();
        assert!(pp.is_warmed_up("price").unwrap());
    }

    #[test]
    fn test_is_warmed_up_nonexistent() {
        let pp = Preprocessing::new();
        assert!(pp.is_warmed_up("nope").is_err());
    }

    // -----------------------------------------------------------------------
    // Global stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats() {
        let mut pp = Preprocessing::new();
        pp.register("price", NormStrategy::None).unwrap();

        pp.process_value("price", 1.0).unwrap();
        pp.process_value("price", 2.0).unwrap();
        pp.tick();

        let stats = pp.stats();
        assert_eq!(stats.total_processed, 2);
        assert_eq!(stats.field_count, 1);
        assert_eq!(stats.ticks, 1);
    }

    #[test]
    fn test_stats_non_finite() {
        let mut pp = Preprocessing::new();
        let cfg =
            FieldConfig::new("price", NormStrategy::None).with_imputation(ImputeStrategy::Zero);
        pp.register_field(cfg).unwrap();

        pp.process_value("price", f64::NAN).unwrap();
        pp.process_value("price", f64::INFINITY).unwrap();

        let stats = pp.stats();
        assert_eq!(stats.total_non_finite, 2);
        assert_eq!(stats.total_imputed, 2);
    }

    // -----------------------------------------------------------------------
    // Tick / Process
    // -----------------------------------------------------------------------

    #[test]
    fn test_tick() {
        let mut pp = Preprocessing::new();
        assert_eq!(pp.current_tick(), 0);
        pp.tick();
        assert_eq!(pp.current_tick(), 1);
    }

    #[test]
    fn test_process() {
        let mut pp = Preprocessing::new();
        assert!(pp.process().is_ok());
        assert_eq!(pp.current_tick(), 1);
    }

    // -----------------------------------------------------------------------
    // Reset
    // -----------------------------------------------------------------------

    #[test]
    fn test_reset_field() {
        let mut pp = Preprocessing::new();
        pp.register("price", NormStrategy::None).unwrap();
        pp.process_value("price", 1.0).unwrap();
        pp.process_value("price", 2.0).unwrap();
        pp.reset_field("price").unwrap();
        let stats = pp.field_stats("price").unwrap();
        assert_eq!(stats.count, 0);
        assert!(stats.last_value.is_none());
    }

    #[test]
    fn test_reset_field_nonexistent() {
        let mut pp = Preprocessing::new();
        assert!(pp.reset_field("nope").is_err());
    }

    #[test]
    fn test_reset_stats() {
        let mut pp = Preprocessing::new();
        pp.register("price", NormStrategy::None).unwrap();
        pp.process_value("price", 1.0).unwrap();
        pp.tick();
        pp.reset_stats();

        let stats = pp.stats();
        assert_eq!(stats.total_processed, 0);
        // Field still registered
        assert_eq!(pp.field_count(), 1);
        // Field stats reset
        let fs = pp.field_stats("price").unwrap();
        assert_eq!(fs.count, 0);
    }

    #[test]
    fn test_reset() {
        let mut pp = Preprocessing::new();
        pp.register("price", NormStrategy::None).unwrap();
        pp.process_value("price", 1.0).unwrap();
        pp.tick();
        pp.reset();
        assert_eq!(pp.field_count(), 0);
        assert_eq!(pp.current_tick(), 0);
    }

    // -----------------------------------------------------------------------
    // WindowBuffer internals
    // -----------------------------------------------------------------------

    #[test]
    fn test_window_buffer_median_even() {
        let mut wb = WindowBuffer::new(10);
        wb.push(1.0);
        wb.push(2.0);
        wb.push(3.0);
        wb.push(4.0);
        let median = wb.median();
        assert!((median - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_window_buffer_median_odd() {
        let mut wb = WindowBuffer::new(10);
        wb.push(1.0);
        wb.push(2.0);
        wb.push(3.0);
        let median = wb.median();
        assert!((median - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_window_buffer_median_empty() {
        let wb = WindowBuffer::new(10);
        assert!((wb.median() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_window_buffer_percentile() {
        let mut wb = WindowBuffer::new(100);
        for i in 0..100 {
            wb.push(i as f64);
        }
        let p25 = wb.percentile(25.0);
        let p75 = wb.percentile(75.0);
        assert!(p25 < p75);
    }

    #[test]
    fn test_window_buffer_eviction() {
        let mut wb = WindowBuffer::new(3);
        wb.push(1.0);
        wb.push(2.0);
        wb.push(3.0);
        wb.push(4.0); // evicts 1.0
        assert_eq!(wb.len(), 3);
        let mean = wb.mean();
        assert!((mean - 3.0).abs() < 1e-10); // (2+3+4)/3 = 3
    }

    #[test]
    fn test_window_buffer_clear() {
        let mut wb = WindowBuffer::new(10);
        wb.push(1.0);
        wb.push(2.0);
        wb.clear();
        assert!(wb.is_empty());
        assert!((wb.mean() - 0.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Running stats internals
    // -----------------------------------------------------------------------

    #[test]
    fn test_running_stats_single() {
        let mut rs = RunningStats::new();
        rs.update(5.0);
        assert!((rs.mean - 5.0).abs() < 1e-10);
        assert!((rs.std_dev() - 0.0).abs() < 1e-10);
        assert!((rs.min - 5.0).abs() < 1e-10);
        assert!((rs.max - 5.0).abs() < 1e-10);
        assert!(rs.last_value.is_some());
    }

    #[test]
    fn test_running_stats_multiple() {
        let mut rs = RunningStats::new();
        rs.update(10.0);
        rs.update(20.0);
        rs.update(30.0);
        assert!((rs.mean - 20.0).abs() < 1e-10);
        assert!(rs.std_dev() > 0.0);
    }

    #[test]
    fn test_running_stats_reset() {
        let mut rs = RunningStats::new();
        rs.update(10.0);
        rs.reset();
        assert_eq!(rs.count, 0);
        assert!((rs.mean - 0.0).abs() < 1e-10);
        assert!(rs.last_value.is_none());
    }

    // -----------------------------------------------------------------------
    // FieldConfig API
    // -----------------------------------------------------------------------

    #[test]
    fn test_field_config_defaults() {
        let fc = FieldConfig::with_defaults("price");
        assert_eq!(fc.name, "price");
        assert_eq!(fc.norm, NormStrategy::None);
        assert!(!fc.outlier_detection);
    }

    #[test]
    fn test_field_config_builders() {
        let fc = FieldConfig::new("price", NormStrategy::ZScore)
            .with_outlier_detection(Some(2.5))
            .with_imputation(ImputeStrategy::Mean);
        assert_eq!(fc.norm, NormStrategy::ZScore);
        assert!(fc.outlier_detection);
        assert_eq!(fc.outlier_z_threshold, Some(2.5));
        assert_eq!(fc.impute, ImputeStrategy::Mean);
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_window_eviction_preserves_std_dev() {
        let cfg = PreprocessingConfig {
            window_size: 5,
            ..Default::default()
        };
        let mut pp = Preprocessing::with_config(cfg).unwrap();
        pp.register("price", NormStrategy::None).unwrap();

        for i in 0..20 {
            pp.process_value("price", i as f64).unwrap();
        }

        let stats = pp.field_stats("price").unwrap();
        assert_eq!(stats.window_size, 5);
        assert!(stats.windowed_std_dev > 0.0);
    }

    #[test]
    fn test_normalisation_not_applied_before_warmup() {
        let cfg = PreprocessingConfig {
            min_observations: 100,
            ..Default::default()
        };
        let mut pp = Preprocessing::with_config(cfg).unwrap();
        pp.register("price", NormStrategy::MinMax).unwrap();

        // Only 5 observations → normalisation not applied → raw pass-through
        for _ in 0..5 {
            pp.process_value("price", 100.0).unwrap();
        }
        let result = pp.process_value("price", 200.0).unwrap();
        assert!((result.value - 200.0).abs() < 1e-10);
    }

    #[test]
    fn test_processed_value_z_score_populated() {
        let cfg = PreprocessingConfig {
            min_observations: 5,
            window_size: 20,
            ..Default::default()
        };
        let mut pp = Preprocessing::with_config(cfg).unwrap();
        let field_cfg = FieldConfig::new("price", NormStrategy::None).with_outlier_detection(None);
        pp.register_field(field_cfg).unwrap();

        for i in 0..20 {
            pp.process_value("price", 100.0 + (i as f64 * 0.5)).unwrap();
        }

        let result = pp.process_value("price", 110.0).unwrap();
        assert!(result.z_score.is_some());
    }

    #[test]
    fn test_multiple_fields_independent() {
        let cfg = PreprocessingConfig {
            min_observations: 2,
            ..Default::default()
        };
        let mut pp = Preprocessing::with_config(cfg).unwrap();
        pp.register("price", NormStrategy::MinMax).unwrap();
        pp.register("volume", NormStrategy::ZScore).unwrap();

        for _ in 0..5 {
            pp.process_value("price", 100.0).unwrap();
            pp.process_value("volume", 1000.0).unwrap();
        }

        let ps = pp.field_stats("price").unwrap();
        let vs = pp.field_stats("volume").unwrap();
        assert_eq!(ps.count, 5);
        assert_eq!(vs.count, 5);
    }

    #[test]
    fn test_total_clamped_stat() {
        let cfg = PreprocessingConfig {
            min_observations: 5,
            window_size: 20,
            outlier_z_threshold: 2.0,
            clamp_outliers: true,
            ..Default::default()
        };
        let mut pp = Preprocessing::with_config(cfg).unwrap();
        let field_cfg = FieldConfig::new("price", NormStrategy::None).with_outlier_detection(None);
        pp.register_field(field_cfg).unwrap();

        for i in 0..20 {
            pp.process_value("price", 100.0 + (i % 3) as f64).unwrap();
        }

        pp.process_value("price", 1_000_000.0).unwrap();

        let stats = pp.stats();
        assert!(stats.total_clamped > 0);
        assert!(stats.total_outliers > 0);
    }
}
