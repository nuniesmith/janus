//! Optimization Module for Strategy Parameter Tuning
//!
//! Provides tools for optimizing trading strategy parameters through:
//! - Grid search parameter optimization
//! - Walk-forward analysis to prevent overfitting
//! - Out-of-sample testing
//! - Performance metric tracking
//!
//! ## Walk-Forward Analysis
//!
//! Walk-forward analysis divides historical data into multiple windows:
//!
//! ```text
//! |-------- In-Sample --------|-- Out-of-Sample --|
//! |     Training/Optimize     |      Validate     |
//!
//! Window 1: [===================][========]
//! Window 2:      [===================][========]
//! Window 3:           [===================][========]
//! Window 4:                [===================][========]
//! ```
//!
//! Each window optimizes on in-sample data and validates on out-of-sample.
//! This helps prevent overfitting and tests parameter stability.

use chrono::{DateTime, Duration, Utc};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;
use tracing::{debug, info};

// Re-export for walk-forward backtest builder pattern
pub use self::WalkForwardBuilder as WalkForwardBacktestBuilder;

/// Errors that can occur during optimization
#[derive(Debug, Error)]
pub enum OptimizationError {
    #[error("Invalid parameter range: {0}")]
    InvalidRange(String),

    #[error("No parameter combinations to test")]
    NoParameterCombinations,

    #[error("Optimization failed: {0}")]
    Failed(String),

    #[error("Walk-forward configuration error: {0}")]
    WalkForwardConfig(String),

    #[error("Insufficient data for walk-forward analysis")]
    InsufficientData,
}

/// A parameter range for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterRange {
    /// Discrete integer values (min, max, step)
    IntRange {
        name: String,
        min: i64,
        max: i64,
        step: i64,
    },
    /// Continuous float values (min, max, step)
    FloatRange {
        name: String,
        min: f64,
        max: f64,
        step: f64,
    },
    /// Specific values to test
    Discrete {
        name: String,
        values: Vec<ParameterValue>,
    },
    /// Boolean parameter
    Boolean { name: String },
}

impl ParameterRange {
    /// Create an integer range parameter
    pub fn int(name: &str, min: i64, max: i64, step: i64) -> Self {
        Self::IntRange {
            name: name.to_string(),
            min,
            max,
            step,
        }
    }

    /// Create a float range parameter
    pub fn float(name: &str, min: f64, max: f64, step: f64) -> Self {
        Self::FloatRange {
            name: name.to_string(),
            min,
            max,
            step,
        }
    }

    /// Create a discrete parameter with specific values
    pub fn discrete(name: &str, values: Vec<ParameterValue>) -> Self {
        Self::Discrete {
            name: name.to_string(),
            values,
        }
    }

    /// Create a boolean parameter
    pub fn boolean(name: &str) -> Self {
        Self::Boolean {
            name: name.to_string(),
        }
    }

    /// Get the parameter name
    pub fn name(&self) -> &str {
        match self {
            Self::IntRange { name, .. } => name,
            Self::FloatRange { name, .. } => name,
            Self::Discrete { name, .. } => name,
            Self::Boolean { name } => name,
        }
    }

    /// Generate all values for this parameter
    pub fn values(&self) -> Vec<ParameterValue> {
        match self {
            Self::IntRange { min, max, step, .. } => {
                let mut values = Vec::new();
                let mut v = *min;
                while v <= *max {
                    values.push(ParameterValue::Int(v));
                    v += step;
                }
                values
            }
            Self::FloatRange { min, max, step, .. } => {
                let mut values = Vec::new();
                let mut v = *min;
                while v <= *max + f64::EPSILON {
                    values.push(ParameterValue::Float(v));
                    v += step;
                }
                values
            }
            Self::Discrete { values, .. } => values.clone(),
            Self::Boolean { .. } => {
                vec![ParameterValue::Bool(false), ParameterValue::Bool(true)]
            }
        }
    }

    /// Get the number of values
    pub fn count(&self) -> usize {
        match self {
            Self::IntRange { min, max, step, .. } => ((max - min) / step + 1) as usize,
            Self::FloatRange { min, max, step, .. } => ((max - min) / step + 1.0) as usize,
            Self::Discrete { values, .. } => values.len(),
            Self::Boolean { .. } => 2,
        }
    }
}

/// A parameter value
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ParameterValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
}

impl std::fmt::Display for ParameterValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Int(v) => write!(f, "{}", v),
            Self::Float(v) => write!(f, "{:.4}", v),
            Self::Bool(v) => write!(f, "{}", v),
            Self::String(v) => write!(f, "{}", v),
        }
    }
}

impl ParameterValue {
    /// Convert to i64 if possible
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Self::Int(v) => Some(*v),
            Self::Float(v) => Some(*v as i64),
            _ => None,
        }
    }

    /// Convert to f64 if possible
    pub fn as_float(&self) -> Option<f64> {
        match self {
            Self::Int(v) => Some(*v as f64),
            Self::Float(v) => Some(*v),
            _ => None,
        }
    }

    /// Convert to bool if possible
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            _ => None,
        }
    }
}

/// A set of parameter values for a single run
pub type ParameterSet = HashMap<String, ParameterValue>;

/// Configuration for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    /// Parameter ranges to optimize
    pub parameters: Vec<ParameterRange>,
    /// Metric to optimize (e.g., "sharpe_ratio", "total_return", "profit_factor")
    pub optimize_metric: OptimizationMetric,
    /// Minimize or maximize the metric
    pub direction: OptimizationDirection,
    /// Maximum number of combinations to test (0 = unlimited)
    pub max_combinations: usize,
    /// Number of top results to keep
    pub top_n: usize,
    /// Enable parallel optimization
    pub parallel: bool,
    /// Maximum number of parallel workers (0 = auto, based on CPU count)
    pub max_workers: usize,
    /// Verbose logging
    pub verbose: bool,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            parameters: Vec::new(),
            optimize_metric: OptimizationMetric::SharpeRatio,
            direction: OptimizationDirection::Maximize,
            max_combinations: 0,
            top_n: 10,
            parallel: true,
            max_workers: 0, // Auto-detect
            verbose: false,
        }
    }
}

impl OptimizationConfig {
    /// Create a new optimization config
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a parameter range
    pub fn with_parameter(mut self, param: ParameterRange) -> Self {
        self.parameters.push(param);
        self
    }

    /// Set the metric to optimize
    pub fn with_metric(mut self, metric: OptimizationMetric) -> Self {
        self.optimize_metric = metric;
        self
    }

    /// Set optimization direction
    pub fn with_direction(mut self, direction: OptimizationDirection) -> Self {
        self.direction = direction;
        self
    }

    /// Set maximum combinations
    pub fn with_max_combinations(mut self, max: usize) -> Self {
        self.max_combinations = max;
        self
    }

    /// Set number of top results to keep
    pub fn with_top_n(mut self, n: usize) -> Self {
        self.top_n = n;
        self
    }

    /// Enable/disable parallel execution
    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    /// Set maximum number of parallel workers
    pub fn with_max_workers(mut self, workers: usize) -> Self {
        self.max_workers = workers;
        self
    }

    /// Get the effective number of workers to use
    pub fn effective_workers(&self) -> usize {
        if self.max_workers == 0 {
            // Auto-detect: use number of CPUs, but cap at 16
            std::thread::available_parallelism()
                .map(|n| n.get().min(16))
                .unwrap_or(4)
        } else {
            self.max_workers
        }
    }

    /// Enable verbose logging
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Calculate total number of parameter combinations
    pub fn total_combinations(&self) -> usize {
        if self.parameters.is_empty() {
            return 0;
        }

        self.parameters.iter().map(|p| p.count()).product()
    }

    /// Generate all parameter combinations
    pub fn generate_combinations(&self) -> Vec<ParameterSet> {
        if self.parameters.is_empty() {
            return Vec::new();
        }

        let mut combinations = vec![ParameterSet::new()];

        for param in &self.parameters {
            let values = param.values();
            let mut new_combinations = Vec::new();

            for combo in combinations {
                for value in &values {
                    let mut new_combo = combo.clone();
                    new_combo.insert(param.name().to_string(), value.clone());
                    new_combinations.push(new_combo);
                }
            }

            combinations = new_combinations;
        }

        // Apply max_combinations limit
        if self.max_combinations > 0 && combinations.len() > self.max_combinations {
            combinations.truncate(self.max_combinations);
        }

        combinations
    }
}

/// Metric to optimize
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationMetric {
    /// Sharpe ratio (risk-adjusted return)
    SharpeRatio,
    /// Total return percentage
    TotalReturn,
    /// Profit factor (gross profit / gross loss)
    ProfitFactor,
    /// Win rate percentage
    WinRate,
    /// Maximum drawdown (minimize)
    MaxDrawdown,
    /// Average trade profit
    AvgProfit,
    /// Number of trades
    TradeCount,
    /// Custom metric by name
    Custom(u32),
}

impl std::fmt::Display for OptimizationMetric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SharpeRatio => write!(f, "sharpe_ratio"),
            Self::TotalReturn => write!(f, "total_return"),
            Self::ProfitFactor => write!(f, "profit_factor"),
            Self::WinRate => write!(f, "win_rate"),
            Self::MaxDrawdown => write!(f, "max_drawdown"),
            Self::AvgProfit => write!(f, "avg_profit"),
            Self::TradeCount => write!(f, "trade_count"),
            Self::Custom(id) => write!(f, "custom_{}", id),
        }
    }
}

/// Optimization direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationDirection {
    Maximize,
    Minimize,
}

/// Result of a single optimization run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRunResult {
    /// Parameter set used
    pub parameters: ParameterSet,
    /// All metrics from the run
    pub metrics: HashMap<String, f64>,
    /// The optimized metric value
    pub metric_value: f64,
    /// Total trades
    pub total_trades: u64,
    /// Win rate
    pub win_rate: f64,
    /// Max drawdown
    pub max_drawdown: f64,
    /// Sharpe ratio
    pub sharpe_ratio: Option<f64>,
    /// Profit factor
    pub profit_factor: Option<f64>,
    /// Run duration in milliseconds
    pub run_duration_ms: u64,
}

impl OptimizationRunResult {
    /// Create a new run result
    pub fn new(parameters: ParameterSet, metric_value: f64) -> Self {
        Self {
            parameters,
            metrics: HashMap::new(),
            metric_value,
            total_trades: 0,
            win_rate: 0.0,
            max_drawdown: 0.0,
            sharpe_ratio: None,
            profit_factor: None,
            run_duration_ms: 0,
        }
    }

    /// Add a metric
    pub fn with_metric(mut self, name: &str, value: f64) -> Self {
        self.metrics.insert(name.to_string(), value);
        self
    }

    /// Get parameter as string for display
    pub fn params_str(&self) -> String {
        self.parameters
            .iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// Overall optimization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    /// Configuration used
    pub metric_optimized: String,
    /// Direction (maximize/minimize)
    pub direction: String,
    /// Total combinations tested
    pub combinations_tested: usize,
    /// Total time taken
    pub total_duration_ms: u64,
    /// Best result
    pub best: Option<OptimizationRunResult>,
    /// Top N results
    pub top_results: Vec<OptimizationRunResult>,
    /// All results (if requested)
    pub all_results: Vec<OptimizationRunResult>,
    /// Start time
    pub start_time: DateTime<Utc>,
    /// End time
    pub end_time: DateTime<Utc>,
}

impl OptimizationResult {
    /// Create a new empty result
    pub fn new(metric: &str, direction: OptimizationDirection) -> Self {
        Self {
            metric_optimized: metric.to_string(),
            direction: format!("{:?}", direction),
            combinations_tested: 0,
            total_duration_ms: 0,
            best: None,
            top_results: Vec::new(),
            all_results: Vec::new(),
            start_time: Utc::now(),
            end_time: Utc::now(),
        }
    }

    /// Get a summary string
    pub fn summary(&self) -> String {
        if let Some(ref best) = self.best {
            format!(
                "Optimization Result: {} {} = {:.4}, tested {} combinations in {}ms\nBest params: {}",
                self.direction,
                self.metric_optimized,
                best.metric_value,
                self.combinations_tested,
                self.total_duration_ms,
                best.params_str()
            )
        } else {
            format!(
                "Optimization Result: No valid results from {} combinations",
                self.combinations_tested
            )
        }
    }
}

/// Configuration for walk-forward analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalkForwardConfig {
    /// Number of walk-forward windows
    pub num_windows: usize,
    /// In-sample period as percentage of window (0.0-1.0)
    pub in_sample_pct: f64,
    /// Minimum number of trades required in each window
    pub min_trades_per_window: usize,
    /// Anchored (expanding) or rolling windows
    pub anchored: bool,
    /// Optimization config for each window
    pub optimization: OptimizationConfig,
}

impl Default for WalkForwardConfig {
    fn default() -> Self {
        Self {
            num_windows: 5,
            in_sample_pct: 0.7,
            min_trades_per_window: 30,
            anchored: false,
            optimization: OptimizationConfig::default(),
        }
    }
}

impl WalkForwardConfig {
    /// Create a new walk-forward config
    pub fn new(num_windows: usize) -> Self {
        Self {
            num_windows,
            ..Default::default()
        }
    }

    /// Set in-sample percentage
    pub fn with_in_sample_pct(mut self, pct: f64) -> Self {
        self.in_sample_pct = pct.clamp(0.1, 0.9);
        self
    }

    /// Set minimum trades per window
    pub fn with_min_trades(mut self, min: usize) -> Self {
        self.min_trades_per_window = min;
        self
    }

    /// Use anchored (expanding) windows
    pub fn anchored(mut self) -> Self {
        self.anchored = true;
        self
    }

    /// Use rolling windows
    pub fn rolling(mut self) -> Self {
        self.anchored = false;
        self
    }

    /// Set optimization config
    pub fn with_optimization(mut self, config: OptimizationConfig) -> Self {
        self.optimization = config;
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), OptimizationError> {
        if self.num_windows < 2 {
            return Err(OptimizationError::WalkForwardConfig(
                "Need at least 2 windows for walk-forward analysis".to_string(),
            ));
        }

        if self.in_sample_pct < 0.1 || self.in_sample_pct > 0.9 {
            return Err(OptimizationError::WalkForwardConfig(
                "in_sample_pct must be between 0.1 and 0.9".to_string(),
            ));
        }

        Ok(())
    }
}

/// A single walk-forward window
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalkForwardWindow {
    /// Window index
    pub index: usize,
    /// In-sample start time
    pub is_start: DateTime<Utc>,
    /// In-sample end time
    pub is_end: DateTime<Utc>,
    /// Out-of-sample start time
    pub oos_start: DateTime<Utc>,
    /// Out-of-sample end time
    pub oos_end: DateTime<Utc>,
    /// Best parameters from in-sample optimization
    pub best_params: Option<ParameterSet>,
    /// In-sample optimization result
    pub in_sample_result: Option<OptimizationRunResult>,
    /// Out-of-sample validation result
    pub out_of_sample_result: Option<OptimizationRunResult>,
}

impl WalkForwardWindow {
    /// Create a new window
    pub fn new(
        index: usize,
        is_start: DateTime<Utc>,
        is_end: DateTime<Utc>,
        oos_start: DateTime<Utc>,
        oos_end: DateTime<Utc>,
    ) -> Self {
        Self {
            index,
            is_start,
            is_end,
            oos_start,
            oos_end,
            best_params: None,
            in_sample_result: None,
            out_of_sample_result: None,
        }
    }

    /// Get in-sample duration
    pub fn in_sample_duration(&self) -> Duration {
        self.is_end - self.is_start
    }

    /// Get out-of-sample duration
    pub fn out_of_sample_duration(&self) -> Duration {
        self.oos_end - self.oos_start
    }
}

/// Walk-forward analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalkForwardResult {
    /// Configuration used
    pub config: WalkForwardConfig,
    /// Individual window results
    pub windows: Vec<WalkForwardWindow>,
    /// Aggregate in-sample metric
    pub avg_in_sample_metric: f64,
    /// Aggregate out-of-sample metric
    pub avg_out_of_sample_metric: f64,
    /// Walk-forward efficiency (OOS / IS performance)
    pub efficiency: f64,
    /// Parameter stability (how consistent were optimal params)
    pub parameter_stability: HashMap<String, f64>,
    /// Total duration
    pub total_duration_ms: u64,
    /// Start time
    pub start_time: DateTime<Utc>,
    /// End time
    pub end_time: DateTime<Utc>,
}

impl WalkForwardResult {
    /// Create a new result
    pub fn new(config: WalkForwardConfig) -> Self {
        Self {
            config,
            windows: Vec::new(),
            avg_in_sample_metric: 0.0,
            avg_out_of_sample_metric: 0.0,
            efficiency: 0.0,
            parameter_stability: HashMap::new(),
            total_duration_ms: 0,
            start_time: Utc::now(),
            end_time: Utc::now(),
        }
    }

    /// Calculate aggregate metrics from windows
    pub fn calculate_aggregates(&mut self) {
        let valid_window_indices: Vec<usize> = self
            .windows
            .iter()
            .enumerate()
            .filter(|(_, w)| w.in_sample_result.is_some() && w.out_of_sample_result.is_some())
            .map(|(i, _)| i)
            .collect();

        if valid_window_indices.is_empty() {
            return;
        }

        let valid_count = valid_window_indices.len();

        // Average in-sample metric
        let is_sum: f64 = valid_window_indices
            .iter()
            .filter_map(|&i| self.windows[i].in_sample_result.as_ref())
            .map(|r| r.metric_value)
            .sum();
        self.avg_in_sample_metric = is_sum / valid_count as f64;

        // Average out-of-sample metric
        let oos_sum: f64 = valid_window_indices
            .iter()
            .filter_map(|&i| self.windows[i].out_of_sample_result.as_ref())
            .map(|r| r.metric_value)
            .sum();
        self.avg_out_of_sample_metric = oos_sum / valid_count as f64;

        // Walk-forward efficiency
        if self.avg_in_sample_metric != 0.0 {
            self.efficiency = self.avg_out_of_sample_metric / self.avg_in_sample_metric;
        }

        // Parameter stability (coefficient of variation for each parameter)
        self.calculate_parameter_stability(&valid_window_indices);
    }

    /// Calculate parameter stability across windows
    fn calculate_parameter_stability(&mut self, window_indices: &[usize]) {
        // Collect all parameter values across windows
        let mut param_values: HashMap<String, Vec<f64>> = HashMap::new();

        for &idx in window_indices {
            if let Some(ref params) = self.windows[idx].best_params {
                for (name, value) in params {
                    if let Some(v) = value.as_float() {
                        param_values
                            .entry(name.clone())
                            .or_insert_with(Vec::new)
                            .push(v);
                    }
                }
            }
        }

        // Calculate coefficient of variation for each parameter
        for (name, values) in param_values {
            if values.len() < 2 {
                continue;
            }

            let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
            if mean == 0.0 {
                continue;
            }

            let variance: f64 =
                values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
            let std_dev = variance.sqrt();
            let cv = std_dev / mean.abs();

            // Stability = 1 - CV (higher is more stable)
            self.parameter_stability.insert(name, (1.0 - cv).max(0.0));
        }
    }

    /// Get a summary string
    pub fn summary(&self) -> String {
        format!(
            "Walk-Forward Result:\n\
             - Windows: {}\n\
             - Avg In-Sample Metric: {:.4}\n\
             - Avg Out-of-Sample Metric: {:.4}\n\
             - Efficiency: {:.2}%\n\
             - Duration: {}ms",
            self.windows.len(),
            self.avg_in_sample_metric,
            self.avg_out_of_sample_metric,
            self.efficiency * 100.0,
            self.total_duration_ms
        )
    }

    /// Check if the strategy passes walk-forward validation
    pub fn is_robust(&self, min_efficiency: f64) -> bool {
        // A strategy is considered robust if:
        // 1. Efficiency is above threshold (OOS performance close to IS)
        // 2. Most parameters are stable across windows

        let stable_params = self
            .parameter_stability
            .values()
            .filter(|&&s| s > 0.5)
            .count();
        let total_params = self.parameter_stability.len();

        let params_stable = total_params == 0 || stable_params > total_params / 2;

        self.efficiency >= min_efficiency && params_stable
    }
}

/// Walk-forward analysis executor
pub struct WalkForwardAnalysis {
    /// Configuration
    config: WalkForwardConfig,
    /// Data start time
    data_start: DateTime<Utc>,
    /// Data end time
    data_end: DateTime<Utc>,
    /// Generated windows
    windows: Vec<WalkForwardWindow>,
}

impl WalkForwardAnalysis {
    /// Create a new walk-forward analysis
    pub fn new(
        config: WalkForwardConfig,
        data_start: DateTime<Utc>,
        data_end: DateTime<Utc>,
    ) -> Result<Self, OptimizationError> {
        config.validate()?;

        let total_duration = data_end - data_start;
        let min_window_duration = Duration::days(7); // At least 1 week per window

        if total_duration < min_window_duration * config.num_windows as i32 {
            return Err(OptimizationError::InsufficientData);
        }

        let mut analysis = Self {
            config,
            data_start,
            data_end,
            windows: Vec::new(),
        };

        analysis.generate_windows();

        Ok(analysis)
    }

    /// Generate walk-forward windows
    fn generate_windows(&mut self) {
        let total_duration = self.data_end - self.data_start;

        if self.config.anchored {
            // Anchored (expanding) windows - IS always starts from the beginning
            let oos_duration = total_duration / (self.config.num_windows as i32 + 1);

            for i in 0..self.config.num_windows {
                let is_start = self.data_start;
                let is_end = self.data_start
                    + total_duration * (i as i32 + 1) / (self.config.num_windows as i32 + 1);

                // Scale IS end by in_sample_pct
                let is_period = is_end - is_start;
                let adjusted_is_end = is_start
                    + Duration::milliseconds(
                        (is_period.num_milliseconds() as f64 * self.config.in_sample_pct) as i64,
                    );

                let oos_start = adjusted_is_end;
                let oos_end = oos_start + oos_duration;

                self.windows.push(WalkForwardWindow::new(
                    i,
                    is_start,
                    adjusted_is_end,
                    oos_start,
                    oos_end.min(self.data_end),
                ));
            }
        } else {
            // Rolling windows
            let window_duration = total_duration / self.config.num_windows as i32;
            let overlap = window_duration / 2; // 50% overlap between windows

            for i in 0..self.config.num_windows {
                let window_start = self.data_start
                    + Duration::milliseconds((overlap.num_milliseconds() as f64 * i as f64) as i64);
                let window_end = (window_start + window_duration).min(self.data_end);

                let is_duration = Duration::milliseconds(
                    ((window_end - window_start).num_milliseconds() as f64
                        * self.config.in_sample_pct) as i64,
                );

                let is_start = window_start;
                let is_end = window_start + is_duration;
                let oos_start = is_end;
                let oos_end = window_end;

                self.windows.push(WalkForwardWindow::new(
                    i, is_start, is_end, oos_start, oos_end,
                ));
            }
        }

        info!(
            "Generated {} walk-forward windows from {} to {}",
            self.windows.len(),
            self.data_start,
            self.data_end
        );
    }

    /// Get the generated windows
    pub fn windows(&self) -> &[WalkForwardWindow] {
        &self.windows
    }

    /// Get configuration
    pub fn config(&self) -> &WalkForwardConfig {
        &self.config
    }

    /// Create an empty result structure for this analysis
    pub fn create_result(&self) -> WalkForwardResult {
        let mut result = WalkForwardResult::new(self.config.clone());
        result.windows = self.windows.clone();
        result
    }
}

// ============================================================================
// Walk-Forward Backtest Runner
// ============================================================================

/// Strategy evaluator trait for walk-forward testing
///
/// Implement this trait for your strategy to enable walk-forward optimization.
pub trait StrategyEvaluator: Send + Sync {
    /// Run the strategy with given parameters on data within the time range.
    /// Returns the optimization run result with computed metrics.
    fn evaluate(
        &self,
        params: &ParameterSet,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
    ) -> Result<OptimizationRunResult, OptimizationError>;
}

/// Walk-forward backtest runner using recorded data
///
/// Executes walk-forward optimization by:
/// 1. Dividing data into in-sample and out-of-sample periods
/// 2. Optimizing parameters on in-sample data
/// 3. Validating on out-of-sample data
/// 4. Aggregating results across all windows
///
/// # Example
///
/// ```rust,ignore
/// use janus_execution::sim::optimization::{
///     WalkForwardBacktestRunner, WalkForwardConfig, OptimizationConfig,
///     ParameterRange, StrategyEvaluator,
/// };
///
/// struct MyStrategy { /* ... */ }
/// impl StrategyEvaluator for MyStrategy {
///     fn evaluate(&self, params: &ParameterSet, start: DateTime<Utc>, end: DateTime<Utc>)
///         -> Result<OptimizationRunResult, OptimizationError> {
///         // Run backtest with params in [start, end] time range
///         // Return results including metric_value
///     }
/// }
///
/// // Configure walk-forward analysis
/// let wf_config = WalkForwardConfig::new(5)
///     .with_in_sample_pct(0.7)
///     .with_optimization(
///         OptimizationConfig::new()
///             .with_parameter(ParameterRange::int("period", 10, 50, 5))
///             .with_metric(OptimizationMetric::SharpeRatio)
///     );
///
/// // Create runner
/// let runner = WalkForwardBacktestRunner::new(wf_config, data_start, data_end)?;
///
/// // Run walk-forward optimization
/// let result = runner.run(&strategy).await?;
///
/// // Check robustness
/// if result.is_robust(0.7) {
///     println!("Strategy passes walk-forward validation!");
/// }
/// ```
pub struct WalkForwardBacktestRunner {
    /// Walk-forward analysis (windows generator)
    analysis: WalkForwardAnalysis,
    /// Whether to run in-sample optimization in parallel
    parallel: bool,
    /// Verbose logging
    verbose: bool,
}

impl WalkForwardBacktestRunner {
    /// Create a new walk-forward backtest runner
    pub fn new(
        config: WalkForwardConfig,
        data_start: DateTime<Utc>,
        data_end: DateTime<Utc>,
    ) -> Result<Self, OptimizationError> {
        let analysis = WalkForwardAnalysis::new(config, data_start, data_end)?;

        Ok(Self {
            analysis,
            parallel: true,
            verbose: false,
        })
    }

    /// Enable/disable parallel optimization
    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    /// Enable verbose logging
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Get the analysis configuration
    pub fn analysis(&self) -> &WalkForwardAnalysis {
        &self.analysis
    }

    /// Run walk-forward optimization with the given strategy evaluator
    ///
    /// This method:
    /// 1. Iterates through each walk-forward window
    /// 2. For each window, optimizes parameters on in-sample data
    /// 3. Validates best parameters on out-of-sample data
    /// 4. Collects and aggregates results
    pub async fn run<S: StrategyEvaluator>(
        &self,
        strategy: &S,
    ) -> Result<WalkForwardResult, OptimizationError> {
        let start = std::time::Instant::now();
        let mut result = self.analysis.create_result();
        result.start_time = Utc::now();

        let opt_config = &self.analysis.config().optimization;

        info!(
            "Starting walk-forward optimization: {} windows, {} parameter combinations",
            self.analysis.windows().len(),
            opt_config.total_combinations()
        );

        // Process each window
        for (window_idx, window) in self.analysis.windows().iter().enumerate() {
            if self.verbose {
                info!(
                    "Window {}: IS [{} - {}], OOS [{} - {}]",
                    window_idx,
                    window.is_start.format("%Y-%m-%d"),
                    window.is_end.format("%Y-%m-%d"),
                    window.oos_start.format("%Y-%m-%d"),
                    window.oos_end.format("%Y-%m-%d"),
                );
            }

            // Run in-sample optimization
            let is_result = self
                .optimize_window(strategy, window.is_start, window.is_end, opt_config)
                .await?;

            if is_result.is_none() {
                info!(
                    "Window {}: No valid in-sample results, skipping",
                    window_idx
                );
                continue;
            }

            let best_is = is_result.unwrap();
            let best_params = best_is.parameters.clone();

            if self.verbose {
                info!(
                    "Window {}: Best IS params: {}, metric: {:.4}",
                    window_idx,
                    best_is.params_str(),
                    best_is.metric_value
                );
            }

            // Run out-of-sample validation with best parameters
            let oos_result = strategy.evaluate(&best_params, window.oos_start, window.oos_end)?;

            if self.verbose {
                info!(
                    "Window {}: OOS metric: {:.4}",
                    window_idx, oos_result.metric_value
                );
            }

            // Update window results
            result.windows[window_idx].best_params = Some(best_params);
            result.windows[window_idx].in_sample_result = Some(best_is);
            result.windows[window_idx].out_of_sample_result = Some(oos_result);
        }

        // Calculate aggregate metrics
        result.calculate_aggregates();
        result.end_time = Utc::now();
        result.total_duration_ms = start.elapsed().as_millis() as u64;

        info!(
            "Walk-forward complete: efficiency={:.2}%, duration={}ms",
            result.efficiency * 100.0,
            result.total_duration_ms
        );

        Ok(result)
    }

    /// Optimize parameters for a single window (in-sample period)
    async fn optimize_window<S: StrategyEvaluator>(
        &self,
        strategy: &S,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
        config: &OptimizationConfig,
    ) -> Result<Option<OptimizationRunResult>, OptimizationError> {
        let combinations = config.generate_combinations();

        if combinations.is_empty() {
            return Err(OptimizationError::NoParameterCombinations);
        }

        let min_trades = self.analysis.config().min_trades_per_window as u64;

        // Choose parallel or sequential evaluation
        let results = if config.parallel && combinations.len() > 1 {
            self.optimize_window_parallel(
                strategy,
                start_time,
                end_time,
                &combinations,
                min_trades,
                config,
            )
            .await?
        } else {
            self.optimize_window_sequential(
                strategy,
                start_time,
                end_time,
                &combinations,
                min_trades,
            )?
        };

        if results.is_empty() {
            return Ok(None);
        }

        // Sort by metric value according to direction
        let mut results = results;
        match config.direction {
            OptimizationDirection::Maximize => {
                results.sort_by(|a, b| {
                    b.metric_value
                        .partial_cmp(&a.metric_value)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            OptimizationDirection::Minimize => {
                results.sort_by(|a, b| {
                    a.metric_value
                        .partial_cmp(&b.metric_value)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }

        Ok(results.into_iter().next())
    }

    /// Sequential optimization (original implementation)
    fn optimize_window_sequential<S: StrategyEvaluator>(
        &self,
        strategy: &S,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
        combinations: &[ParameterSet],
        min_trades: u64,
    ) -> Result<Vec<OptimizationRunResult>, OptimizationError> {
        let mut results: Vec<OptimizationRunResult> = Vec::with_capacity(combinations.len());

        for params in combinations {
            match strategy.evaluate(params, start_time, end_time) {
                Ok(result) => {
                    if result.total_trades >= min_trades {
                        results.push(result);
                    }
                }
                Err(e) => {
                    if self.verbose {
                        debug!("Parameter combination failed: {:?}", e);
                    }
                }
            }
        }

        Ok(results)
    }

    /// Parallel optimization using tokio tasks
    async fn optimize_window_parallel<S: StrategyEvaluator>(
        &self,
        strategy: &S,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
        combinations: &[ParameterSet],
        min_trades: u64,
        config: &OptimizationConfig,
    ) -> Result<Vec<OptimizationRunResult>, OptimizationError> {
        debug!(
            "Running parallel optimization with {} workers for {} combinations",
            config.effective_workers(),
            combinations.len()
        );

        // Delegate to the synchronous rayon-based parallel implementation.
        // This avoids the previous unsafe raw-pointer casting that was used
        // to bypass Send bounds — rayon's scoped parallelism guarantees the
        // borrowed `strategy` reference outlives all worker threads.
        let final_results = self.optimize_window_sync_parallel(
            strategy,
            start_time,
            end_time,
            combinations,
            min_trades,
            config,
        );

        debug!(
            "Parallel optimization complete: {} valid results",
            final_results.len()
        );

        Ok(final_results)
    }

    /// Run walk-forward optimization synchronously (blocking)
    pub fn run_sync<S: StrategyEvaluator>(
        &self,
        strategy: &S,
    ) -> Result<WalkForwardResult, OptimizationError> {
        let start = std::time::Instant::now();
        let mut result = self.analysis.create_result();
        result.start_time = Utc::now();

        let opt_config = &self.analysis.config().optimization;

        info!(
            "Starting walk-forward optimization (sync): {} windows, {} combinations",
            self.analysis.windows().len(),
            opt_config.total_combinations()
        );

        for (window_idx, window) in self.analysis.windows().iter().enumerate() {
            if self.verbose {
                info!(
                    "Window {}: IS [{} - {}], OOS [{} - {}]",
                    window_idx,
                    window.is_start.format("%Y-%m-%d"),
                    window.is_end.format("%Y-%m-%d"),
                    window.oos_start.format("%Y-%m-%d"),
                    window.oos_end.format("%Y-%m-%d"),
                );
            }

            // In-sample optimization
            let is_result =
                self.optimize_window_sync(strategy, window.is_start, window.is_end, opt_config)?;

            if is_result.is_none() {
                info!("Window {}: No valid IS results", window_idx);
                continue;
            }

            let best_is = is_result.unwrap();
            let best_params = best_is.parameters.clone();

            // Out-of-sample validation
            let oos_result = strategy.evaluate(&best_params, window.oos_start, window.oos_end)?;

            // Store results
            result.windows[window_idx].best_params = Some(best_params);
            result.windows[window_idx].in_sample_result = Some(best_is);
            result.windows[window_idx].out_of_sample_result = Some(oos_result);
        }

        result.calculate_aggregates();
        result.end_time = Utc::now();
        result.total_duration_ms = start.elapsed().as_millis() as u64;

        Ok(result)
    }

    /// Synchronous window optimization
    fn optimize_window_sync<S: StrategyEvaluator>(
        &self,
        strategy: &S,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
        config: &OptimizationConfig,
    ) -> Result<Option<OptimizationRunResult>, OptimizationError> {
        let combinations = config.generate_combinations();

        if combinations.is_empty() {
            return Err(OptimizationError::NoParameterCombinations);
        }

        let min_trades = self.analysis.config().min_trades_per_window as u64;

        // Choose parallel or sequential evaluation
        let results = if config.parallel && combinations.len() > 1 {
            self.optimize_window_sync_parallel(
                strategy,
                start_time,
                end_time,
                &combinations,
                min_trades,
                config,
            )
        } else {
            self.optimize_window_sync_sequential(
                strategy,
                start_time,
                end_time,
                &combinations,
                min_trades,
            )
        };

        if results.is_empty() {
            return Ok(None);
        }

        // Sort by metric
        let mut results = results;
        match config.direction {
            OptimizationDirection::Maximize => {
                results.sort_by(|a, b| {
                    b.metric_value
                        .partial_cmp(&a.metric_value)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            OptimizationDirection::Minimize => {
                results.sort_by(|a, b| {
                    a.metric_value
                        .partial_cmp(&b.metric_value)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }

        Ok(results.into_iter().next())
    }

    /// Sequential synchronous optimization
    fn optimize_window_sync_sequential<S: StrategyEvaluator>(
        &self,
        strategy: &S,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
        combinations: &[ParameterSet],
        min_trades: u64,
    ) -> Vec<OptimizationRunResult> {
        let mut results = Vec::with_capacity(combinations.len());

        for params in combinations {
            if let Ok(result) = strategy.evaluate(params, start_time, end_time) {
                if result.total_trades >= min_trades {
                    results.push(result);
                }
            }
        }

        results
    }

    /// Parallel synchronous optimization using Rayon
    fn optimize_window_sync_parallel<S: StrategyEvaluator>(
        &self,
        strategy: &S,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
        combinations: &[ParameterSet],
        min_trades: u64,
        config: &OptimizationConfig,
    ) -> Vec<OptimizationRunResult> {
        let num_workers = config.effective_workers();
        let verbose = self.verbose;

        debug!(
            "Running parallel sync optimization with {} workers for {} combinations",
            num_workers,
            combinations.len()
        );

        // Configure Rayon thread pool if max_workers is specified
        let pool = if num_workers > 0 && num_workers < rayon::current_num_threads() {
            Some(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(num_workers)
                    .build()
                    .ok(),
            )
        } else {
            None
        };

        let evaluate_combinations = || {
            combinations
                .par_iter()
                .filter_map(
                    |params| match strategy.evaluate(params, start_time, end_time) {
                        Ok(result) => {
                            if result.total_trades >= min_trades {
                                Some(result)
                            } else {
                                None
                            }
                        }
                        Err(e) => {
                            if verbose {
                                debug!("Parameter combination failed: {:?}", e);
                            }
                            None
                        }
                    },
                )
                .collect::<Vec<_>>()
        };

        let results = match pool {
            Some(Some(ref pool)) => pool.install(evaluate_combinations),
            _ => evaluate_combinations(),
        };

        debug!(
            "Parallel sync optimization complete: {} valid results",
            results.len()
        );

        results
    }
}

/// Builder for creating walk-forward backtest configurations
pub struct WalkForwardBuilder {
    num_windows: usize,
    in_sample_pct: f64,
    min_trades: usize,
    anchored: bool,
    parameters: Vec<ParameterRange>,
    metric: OptimizationMetric,
    direction: OptimizationDirection,
    parallel: bool,
    max_workers: usize,
    verbose: bool,
}

impl Default for WalkForwardBuilder {
    fn default() -> Self {
        Self {
            num_windows: 5,
            in_sample_pct: 0.7,
            min_trades: 10,
            anchored: false,
            parameters: Vec::new(),
            metric: OptimizationMetric::SharpeRatio,
            direction: OptimizationDirection::Maximize,
            parallel: true,
            max_workers: 0,
            verbose: false,
        }
    }
}

impl WalkForwardBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set number of walk-forward windows
    pub fn windows(mut self, n: usize) -> Self {
        self.num_windows = n;
        self
    }

    /// Set in-sample percentage (0.0-1.0)
    pub fn in_sample_pct(mut self, pct: f64) -> Self {
        self.in_sample_pct = pct.clamp(0.1, 0.9);
        self
    }

    /// Set minimum trades per window
    pub fn min_trades(mut self, n: usize) -> Self {
        self.min_trades = n;
        self
    }

    /// Use anchored (expanding) windows
    pub fn anchored(mut self) -> Self {
        self.anchored = true;
        self
    }

    /// Use rolling windows
    pub fn rolling(mut self) -> Self {
        self.anchored = false;
        self
    }

    /// Add a parameter to optimize
    pub fn parameter(mut self, param: ParameterRange) -> Self {
        self.parameters.push(param);
        self
    }

    /// Set the optimization metric
    pub fn metric(mut self, metric: OptimizationMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Set optimization direction
    pub fn direction(mut self, dir: OptimizationDirection) -> Self {
        self.direction = dir;
        self
    }

    /// Enable parallel execution
    pub fn parallel(mut self, p: bool) -> Self {
        self.parallel = p;
        self
    }

    /// Set maximum parallel workers (0 = auto-detect based on CPU count)
    pub fn max_workers(mut self, n: usize) -> Self {
        self.max_workers = n;
        self
    }

    /// Enable verbose logging
    pub fn verbose(mut self, v: bool) -> Self {
        self.verbose = v;
        self
    }

    /// Build the walk-forward runner
    pub fn build(
        self,
        data_start: DateTime<Utc>,
        data_end: DateTime<Utc>,
    ) -> Result<WalkForwardBacktestRunner, OptimizationError> {
        let opt_config = OptimizationConfig {
            parameters: self.parameters,
            optimize_metric: self.metric,
            direction: self.direction,
            max_combinations: 0,
            top_n: 10,
            parallel: self.parallel,
            max_workers: self.max_workers,
            verbose: self.verbose,
        };

        let wf_config = WalkForwardConfig {
            num_windows: self.num_windows,
            in_sample_pct: self.in_sample_pct,
            min_trades_per_window: self.min_trades,
            anchored: self.anchored,
            optimization: opt_config,
        };

        WalkForwardBacktestRunner::new(wf_config, data_start, data_end)
            .map(|r| r.with_parallel(self.parallel).with_verbose(self.verbose))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter_range_int() {
        let param = ParameterRange::int("fast_period", 5, 20, 5);

        assert_eq!(param.name(), "fast_period");
        assert_eq!(param.count(), 4); // 5, 10, 15, 20

        let values = param.values();
        assert_eq!(values.len(), 4);
        assert_eq!(values[0], ParameterValue::Int(5));
        assert_eq!(values[3], ParameterValue::Int(20));
    }

    #[test]
    fn test_parameter_range_float() {
        let param = ParameterRange::float("threshold", 0.0, 1.0, 0.25);

        assert_eq!(param.count(), 5); // 0.0, 0.25, 0.5, 0.75, 1.0

        let values = param.values();
        assert_eq!(values.len(), 5);
    }

    #[test]
    fn test_parameter_range_boolean() {
        let param = ParameterRange::boolean("use_atr");

        assert_eq!(param.count(), 2);

        let values = param.values();
        assert_eq!(values[0], ParameterValue::Bool(false));
        assert_eq!(values[1], ParameterValue::Bool(true));
    }

    #[test]
    fn test_parameter_value_conversion() {
        let int_val = ParameterValue::Int(42);
        assert_eq!(int_val.as_int(), Some(42));
        assert_eq!(int_val.as_float(), Some(42.0));

        let float_val = ParameterValue::Float(2.72);
        assert_eq!(float_val.as_float(), Some(2.72));
        assert_eq!(float_val.as_int(), Some(2));

        let bool_val = ParameterValue::Bool(true);
        assert_eq!(bool_val.as_bool(), Some(true));
    }

    #[test]
    fn test_optimization_config_combinations() {
        let config = OptimizationConfig::new()
            .with_parameter(ParameterRange::int("fast", 5, 10, 5)) // 2 values
            .with_parameter(ParameterRange::int("slow", 20, 30, 10)); // 2 values

        assert_eq!(config.total_combinations(), 4);

        let combinations = config.generate_combinations();
        assert_eq!(combinations.len(), 4);

        // Verify all combinations exist
        let has_5_20 = combinations.iter().any(|c| {
            c.get("fast") == Some(&ParameterValue::Int(5))
                && c.get("slow") == Some(&ParameterValue::Int(20))
        });
        assert!(has_5_20);
    }

    #[test]
    fn test_optimization_config_max_combinations() {
        let config = OptimizationConfig::new()
            .with_parameter(ParameterRange::int("a", 1, 10, 1)) // 10 values
            .with_parameter(ParameterRange::int("b", 1, 10, 1)) // 10 values
            .with_max_combinations(20);

        assert_eq!(config.total_combinations(), 100);

        let combinations = config.generate_combinations();
        assert_eq!(combinations.len(), 20); // Limited to 20
    }

    #[test]
    fn test_optimization_result() {
        let result = OptimizationRunResult::new(ParameterSet::new(), 1.5)
            .with_metric("sharpe", 1.5)
            .with_metric("return", 25.0);

        assert_eq!(result.metric_value, 1.5);
        assert_eq!(result.metrics.get("sharpe"), Some(&1.5));
        assert_eq!(result.metrics.get("return"), Some(&25.0));
    }

    #[test]
    fn test_walk_forward_config() {
        let config = WalkForwardConfig::new(5)
            .with_in_sample_pct(0.8)
            .with_min_trades(50)
            .rolling();

        assert_eq!(config.num_windows, 5);
        assert_eq!(config.in_sample_pct, 0.8);
        assert_eq!(config.min_trades_per_window, 50);
        assert!(!config.anchored);

        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_walk_forward_config_validation() {
        let config = WalkForwardConfig::new(1); // Too few windows
        assert!(config.validate().is_err());

        let mut config2 = WalkForwardConfig::new(5);
        config2.in_sample_pct = 0.95; // Too high
        assert!(config2.validate().is_err());
    }

    #[test]
    fn test_walk_forward_analysis_windows() {
        let start = Utc::now() - Duration::days(100);
        let end = Utc::now();

        let config = WalkForwardConfig::new(4).with_in_sample_pct(0.7);

        let analysis = WalkForwardAnalysis::new(config, start, end).unwrap();

        assert_eq!(analysis.windows().len(), 4);

        // Each window should have proper ordering
        for window in analysis.windows() {
            assert!(window.is_start < window.is_end);
            assert!(window.is_end <= window.oos_start);
            assert!(window.oos_start < window.oos_end);
        }
    }

    #[test]
    fn test_walk_forward_result_robustness() {
        let config = WalkForwardConfig::new(4);
        let mut result = WalkForwardResult::new(config);

        result.avg_in_sample_metric = 1.5;
        result.avg_out_of_sample_metric = 1.2;
        result.efficiency = 0.8;

        // With 80% efficiency, should pass 70% threshold
        assert!(result.is_robust(0.7));
        // But not 90%
        assert!(!result.is_robust(0.9));
    }

    #[test]
    fn test_optimization_metric_display() {
        assert_eq!(
            format!("{}", OptimizationMetric::SharpeRatio),
            "sharpe_ratio"
        );
        assert_eq!(
            format!("{}", OptimizationMetric::TotalReturn),
            "total_return"
        );
        assert_eq!(
            format!("{}", OptimizationMetric::MaxDrawdown),
            "max_drawdown"
        );
    }

    #[test]
    fn test_walk_forward_window() {
        let now = Utc::now();
        let window = WalkForwardWindow::new(
            0,
            now,
            now + Duration::days(70),
            now + Duration::days(70),
            now + Duration::days(100),
        );

        assert_eq!(window.index, 0);
        assert_eq!(window.in_sample_duration(), Duration::days(70));
        assert_eq!(window.out_of_sample_duration(), Duration::days(30));
    }

    #[test]
    fn test_insufficient_data() {
        let start = Utc::now();
        let end = start + Duration::days(10); // Only 10 days

        let config = WalkForwardConfig::new(10); // 10 windows for 10 days = not enough

        let result = WalkForwardAnalysis::new(config, start, end);
        assert!(result.is_err());
    }
}
