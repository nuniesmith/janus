//! Parameter Constraints and Search Space
//!
//! This module defines the search space for hyperparameter optimization,
//! including parameter bounds, constraints, and asset-specific adjustments.
//!
//! The search space enforces floor values to prevent whipsaw trades while
//! allowing the optimizer to explore within safe boundaries.

use crate::asset::{AssetCategory, AssetConfig};
use crate::error::{OptimizerError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Minimum allowed EMA fast period (to prevent noise trading)
pub const MIN_EMA_FAST_PERIOD: u32 = 8;

/// Minimum allowed EMA slow period
pub const MIN_EMA_SLOW_PERIOD: u32 = 21;

/// Minimum gap between fast and slow EMA periods
pub const MIN_EMA_PERIOD_GAP: u32 = 10;

/// Minimum profit floor to cover fees and slippage
pub const MIN_PROFIT_FLOOR: f64 = 0.40;

/// A single parameter bound definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterBounds {
    /// Minimum value (inclusive)
    pub min: f64,

    /// Maximum value (inclusive)
    pub max: f64,

    /// Step size for discrete parameters (None for continuous)
    pub step: Option<f64>,

    /// Whether this parameter is an integer
    pub is_integer: bool,

    /// Floor value that cannot be optimized below (for safety constraints)
    pub floor: Option<f64>,

    /// Ceiling value that cannot be optimized above
    pub ceiling: Option<f64>,
}

impl ParameterBounds {
    /// Create new continuous parameter bounds
    pub fn continuous(min: f64, max: f64) -> Self {
        Self {
            min,
            max,
            step: None,
            is_integer: false,
            floor: None,
            ceiling: None,
        }
    }

    /// Create new discrete parameter bounds with step
    pub fn discrete(min: f64, max: f64, step: f64) -> Self {
        Self {
            min,
            max,
            step: Some(step),
            is_integer: false,
            floor: None,
            ceiling: None,
        }
    }

    /// Create new integer parameter bounds
    pub fn integer(min: i32, max: i32) -> Self {
        Self {
            min: min as f64,
            max: max as f64,
            step: Some(1.0),
            is_integer: true,
            floor: None,
            ceiling: None,
        }
    }

    /// Create boolean parameter (0 or 1)
    pub fn boolean() -> Self {
        Self {
            min: 0.0,
            max: 1.0,
            step: Some(1.0),
            is_integer: true,
            floor: None,
            ceiling: None,
        }
    }

    /// Set a floor value (minimum that the optimizer can explore)
    pub fn with_floor(mut self, floor: f64) -> Self {
        self.floor = Some(floor);
        // Adjust min if it's below floor
        if self.min < floor {
            self.min = floor;
        }
        self
    }

    /// Set a ceiling value
    pub fn with_ceiling(mut self, ceiling: f64) -> Self {
        self.ceiling = Some(ceiling);
        // Adjust max if it's above ceiling
        if self.max > ceiling {
            self.max = ceiling;
        }
        self
    }

    /// Apply floor constraint, adjusting min if necessary
    pub fn apply_floor(&mut self, floor: f64) {
        self.floor = Some(floor);
        if self.min < floor {
            self.min = floor;
        }
    }

    /// Validate that a value is within bounds
    pub fn validate(&self, value: f64) -> Result<f64> {
        // Check floor
        if let Some(floor) = self.floor
            && value < floor
        {
            return Err(OptimizerError::ConstraintViolation(format!(
                "Value {} is below floor {}",
                value, floor
            )));
        }

        // Check ceiling
        if let Some(ceiling) = self.ceiling
            && value > ceiling
        {
            return Err(OptimizerError::ConstraintViolation(format!(
                "Value {} is above ceiling {}",
                value, ceiling
            )));
        }

        // Check bounds
        if value < self.min || value > self.max {
            return Err(OptimizerError::ParameterOutOfBounds {
                name: "parameter".to_string(),
                value,
                min: self.min,
                max: self.max,
            });
        }

        // Snap to step if discrete
        if let Some(step) = self.step {
            let snapped = ((value - self.min) / step).round() * step + self.min;
            return Ok(snapped.clamp(self.min, self.max));
        }

        Ok(value)
    }

    /// Clamp a value to be within bounds
    pub fn clamp(&self, value: f64) -> f64 {
        let effective_min = self.floor.unwrap_or(self.min).max(self.min);
        let effective_max = self.ceiling.unwrap_or(self.max).min(self.max);

        let clamped = value.clamp(effective_min, effective_max);

        // Snap to step if discrete
        if let Some(step) = self.step {
            let snapped = ((clamped - effective_min) / step).round() * step + effective_min;
            snapped.clamp(effective_min, effective_max)
        } else {
            clamped
        }
    }

    /// Get the effective minimum (considering floor)
    pub fn effective_min(&self) -> f64 {
        self.floor.unwrap_or(self.min).max(self.min)
    }

    /// Get the effective maximum (considering ceiling)
    pub fn effective_max(&self) -> f64 {
        self.ceiling.unwrap_or(self.max).min(self.max)
    }

    /// Get the range size
    pub fn range(&self) -> f64 {
        self.effective_max() - self.effective_min()
    }

    /// Get the number of discrete values (if applicable)
    pub fn num_values(&self) -> Option<usize> {
        self.step.map(|s| ((self.range() / s).floor() as usize) + 1)
    }
}

/// A constraint function that can be applied to parameters
#[derive(Debug, Clone)]
pub struct Constraint {
    /// Name of the constraint
    pub name: String,

    /// Parameters involved in this constraint
    pub parameters: Vec<String>,

    /// Constraint type
    pub constraint_type: ConstraintType,
}

/// Types of constraints
#[derive(Debug, Clone)]
pub enum ConstraintType {
    /// One parameter must be less than another by at least a margin
    LessThanWithMargin {
        less: String,
        greater: String,
        margin: f64,
    },

    /// Parameter must be greater than or equal to a floor value
    Floor { parameter: String, value: f64 },

    /// Sum of parameters must be less than or equal to a value
    SumLessThanOrEqual { parameters: Vec<String>, max: f64 },

    /// Custom constraint with validation function name
    Custom { name: String },
}

impl Constraint {
    /// Create an EMA period gap constraint
    pub fn ema_period_gap(fast_param: &str, slow_param: &str, min_gap: u32) -> Self {
        Self {
            name: "ema_period_gap".to_string(),
            parameters: vec![fast_param.to_string(), slow_param.to_string()],
            constraint_type: ConstraintType::LessThanWithMargin {
                less: fast_param.to_string(),
                greater: slow_param.to_string(),
                margin: min_gap as f64,
            },
        }
    }

    /// Create a floor constraint
    pub fn floor(parameter: &str, value: f64) -> Self {
        Self {
            name: format!("{}_floor", parameter),
            parameters: vec![parameter.to_string()],
            constraint_type: ConstraintType::Floor {
                parameter: parameter.to_string(),
                value,
            },
        }
    }

    /// Validate a set of parameters against this constraint
    pub fn validate(&self, params: &HashMap<String, f64>) -> Result<()> {
        match &self.constraint_type {
            ConstraintType::LessThanWithMargin {
                less,
                greater,
                margin,
            } => {
                let less_val =
                    params
                        .get(less)
                        .ok_or_else(|| OptimizerError::InvalidParameter {
                            name: less.clone(),
                            reason: "Parameter not found".to_string(),
                        })?;
                let greater_val =
                    params
                        .get(greater)
                        .ok_or_else(|| OptimizerError::InvalidParameter {
                            name: greater.clone(),
                            reason: "Parameter not found".to_string(),
                        })?;

                if less_val + margin > *greater_val {
                    return Err(OptimizerError::ConstraintViolation(format!(
                        "{} ({}) + {} must be <= {} ({})",
                        less, less_val, margin, greater, greater_val
                    )));
                }
            }
            ConstraintType::Floor { parameter, value } => {
                let param_val =
                    params
                        .get(parameter)
                        .ok_or_else(|| OptimizerError::InvalidParameter {
                            name: parameter.clone(),
                            reason: "Parameter not found".to_string(),
                        })?;

                if param_val < value {
                    return Err(OptimizerError::ConstraintViolation(format!(
                        "{} ({}) must be >= {}",
                        parameter, param_val, value
                    )));
                }
            }
            ConstraintType::SumLessThanOrEqual { parameters, max } => {
                let sum: f64 = parameters.iter().filter_map(|p| params.get(p)).sum();

                if sum > *max {
                    return Err(OptimizerError::ConstraintViolation(format!(
                        "Sum of {:?} ({}) must be <= {}",
                        parameters, sum, max
                    )));
                }
            }
            ConstraintType::Custom { name } => {
                // Custom constraints would be handled by external validation
                tracing::debug!(constraint = name, "Custom constraint validation skipped");
            }
        }
        Ok(())
    }
}

/// Search space definition for optimization
#[derive(Debug, Clone)]
pub struct SearchSpace {
    /// Parameter bounds by name
    parameters: HashMap<String, ParameterBounds>,

    /// Constraints to enforce
    constraints: Vec<Constraint>,

    /// Asset category (for applying category-specific constraints)
    asset_category: Option<AssetCategory>,
}

impl SearchSpace {
    /// Create a new empty search space
    pub fn new() -> Self {
        Self {
            parameters: HashMap::new(),
            constraints: Vec::new(),
            asset_category: None,
        }
    }

    /// Create a search space with default EMA strategy parameters
    pub fn ema_strategy() -> Self {
        let mut space = Self::new();

        // EMA periods
        space.add_parameter(
            "ema_fast_period",
            ParameterBounds::integer(MIN_EMA_FAST_PERIOD as i32, 14),
        );
        space.add_parameter(
            "ema_slow_period",
            ParameterBounds::integer(MIN_EMA_SLOW_PERIOD as i32, 40),
        );

        // ATR parameters
        space.add_parameter("atr_length", ParameterBounds::integer(10, 21));
        space.add_parameter("atr_multiplier", ParameterBounds::discrete(1.5, 3.5, 0.25));

        // Stop and profit parameters
        space.add_parameter(
            "min_trailing_stop_pct",
            ParameterBounds::discrete(0.5, 2.0, 0.1),
        );
        space.add_parameter(
            "min_ema_spread_pct",
            ParameterBounds::discrete(0.15, 0.50, 0.02),
        );
        space.add_parameter(
            "min_profit_pct",
            ParameterBounds::discrete(MIN_PROFIT_FLOOR, 1.0, 0.1),
        );
        space.add_parameter("take_profit_pct", ParameterBounds::discrete(3.0, 15.0, 1.0));

        // Higher timeframe settings
        space.add_parameter("require_htf_alignment", ParameterBounds::boolean());
        space.add_parameter(
            "htf_timeframe_minutes",
            ParameterBounds::discrete(15.0, 60.0, 15.0),
        );

        // Cooldown
        space.add_parameter("cooldown_bars", ParameterBounds::integer(3, 10));

        // Add EMA period gap constraint
        space.add_constraint(Constraint::ema_period_gap(
            "ema_fast_period",
            "ema_slow_period",
            MIN_EMA_PERIOD_GAP,
        ));

        space
    }

    /// Create a search space for a specific asset with appropriate constraints
    pub fn for_asset(asset: &str, category: AssetCategory) -> Self {
        let mut space = Self::ema_strategy();
        space.asset_category = Some(category);

        // Apply category-specific floors
        let min_spread = category.min_ema_spread_pct();
        let (atr_min, atr_max) = category.atr_multiplier_range();
        let (tp_min, tp_max) = category.take_profit_range();
        let min_profit = category.min_profit_floor();

        // Update parameter bounds based on category
        if let Some(bounds) = space.parameters.get_mut("min_ema_spread_pct") {
            bounds.apply_floor(min_spread);
            // Extend max for meme coins
            if category == AssetCategory::Meme {
                bounds.max = 0.60;
            }
        }

        if let Some(bounds) = space.parameters.get_mut("atr_multiplier") {
            bounds.min = atr_min;
            bounds.max = atr_max;
        }

        if let Some(bounds) = space.parameters.get_mut("take_profit_pct") {
            bounds.min = tp_min;
            bounds.max = tp_max;
        }

        if let Some(bounds) = space.parameters.get_mut("min_profit_pct") {
            bounds.apply_floor(min_profit);
        }

        // Add floor constraint for EMA spread
        space.add_constraint(Constraint::floor("min_ema_spread_pct", min_spread));

        tracing::debug!(
            asset = asset,
            category = %category,
            min_spread = min_spread,
            atr_range = ?(atr_min, atr_max),
            tp_range = ?(tp_min, tp_max),
            "Created search space for asset"
        );

        space
    }

    /// Apply asset-specific constraints to the search space
    pub fn with_asset_constraints(mut self, asset_config: &AssetConfig) -> Self {
        self.asset_category = Some(asset_config.category);

        // Apply custom overrides if present
        if let Some(custom_spread) = asset_config.custom_min_ema_spread {
            if let Some(bounds) = self.parameters.get_mut("min_ema_spread_pct") {
                bounds.apply_floor(custom_spread);
            }
            self.add_constraint(Constraint::floor("min_ema_spread_pct", custom_spread));
        } else {
            let min_spread = asset_config.category.min_ema_spread_pct();
            if let Some(bounds) = self.parameters.get_mut("min_ema_spread_pct") {
                bounds.apply_floor(min_spread);
            }
        }

        self
    }

    /// Add a parameter to the search space
    pub fn add_parameter(&mut self, name: impl Into<String>, bounds: ParameterBounds) {
        self.parameters.insert(name.into(), bounds);
    }

    /// Add a constraint to the search space
    pub fn add_constraint(&mut self, constraint: Constraint) {
        self.constraints.push(constraint);
    }

    /// Get parameter bounds by name
    pub fn get(&self, name: &str) -> Option<&ParameterBounds> {
        self.parameters.get(name)
    }

    /// Get mutable parameter bounds by name
    pub fn get_mut(&mut self, name: &str) -> Option<&mut ParameterBounds> {
        self.parameters.get_mut(name)
    }

    /// Get all parameter names
    pub fn parameter_names(&self) -> Vec<&String> {
        self.parameters.keys().collect()
    }

    /// Get all parameters
    pub fn parameters(&self) -> &HashMap<String, ParameterBounds> {
        &self.parameters
    }

    /// Get all constraints
    pub fn constraints(&self) -> &[Constraint] {
        &self.constraints
    }

    /// Get the number of parameters
    pub fn num_parameters(&self) -> usize {
        self.parameters.len()
    }

    /// Validate a set of sampled parameters against all constraints
    pub fn validate(&self, params: &HashMap<String, f64>) -> Result<()> {
        // Check each parameter is within bounds
        for (name, value) in params {
            if let Some(bounds) = self.parameters.get(name) {
                bounds
                    .validate(*value)
                    .map_err(|_| OptimizerError::ParameterOutOfBounds {
                        name: name.clone(),
                        value: *value,
                        min: bounds.effective_min(),
                        max: bounds.effective_max(),
                    })?;
            }
        }

        // Check all constraints
        for constraint in &self.constraints {
            constraint.validate(params)?;
        }

        Ok(())
    }

    /// Clamp all parameters to be within bounds
    pub fn clamp_all(&self, params: &mut HashMap<String, f64>) {
        for (name, value) in params.iter_mut() {
            if let Some(bounds) = self.parameters.get(name) {
                *value = bounds.clamp(*value);
            }
        }
    }

    /// Check if search space is valid (has parameters and consistent bounds)
    pub fn is_valid(&self) -> bool {
        if self.parameters.is_empty() {
            return false;
        }

        for bounds in self.parameters.values() {
            if bounds.effective_min() > bounds.effective_max() {
                return false;
            }
        }

        true
    }
}

impl Default for SearchSpace {
    fn default() -> Self {
        Self::ema_strategy()
    }
}

/// Sampled parameters from the search space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampledParams {
    /// Parameter values
    pub values: HashMap<String, f64>,

    /// Whether parameters were validated
    pub validated: bool,
}

impl SampledParams {
    /// Create new sampled params
    pub fn new(values: HashMap<String, f64>) -> Self {
        Self {
            values,
            validated: false,
        }
    }

    /// Get a parameter value
    pub fn get(&self, name: &str) -> Option<f64> {
        self.values.get(name).copied()
    }

    /// Get a parameter as integer
    pub fn get_int(&self, name: &str) -> Option<i32> {
        self.values.get(name).map(|v| *v as i32)
    }

    /// Get a parameter as boolean
    pub fn get_bool(&self, name: &str) -> Option<bool> {
        self.values.get(name).map(|v| *v >= 0.5)
    }

    /// Mark as validated
    pub fn mark_validated(&mut self) {
        self.validated = true;
    }

    /// Get all parameter names
    pub fn names(&self) -> Vec<&String> {
        self.values.keys().collect()
    }
}

impl std::ops::Index<&str> for SampledParams {
    type Output = f64;

    fn index(&self, name: &str) -> &Self::Output {
        self.values.get(name).expect("Parameter not found")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter_bounds_continuous() {
        let bounds = ParameterBounds::continuous(0.0, 1.0);
        assert_eq!(bounds.validate(0.5).unwrap(), 0.5);
        assert!(bounds.validate(-0.1).is_err());
        assert!(bounds.validate(1.1).is_err());
    }

    #[test]
    fn test_parameter_bounds_discrete() {
        let bounds = ParameterBounds::discrete(0.0, 1.0, 0.25);
        assert_eq!(bounds.validate(0.3).unwrap(), 0.25); // Snapped to 0.25
        assert_eq!(bounds.validate(0.4).unwrap(), 0.5); // Snapped to 0.5
    }

    #[test]
    fn test_parameter_bounds_integer() {
        let bounds = ParameterBounds::integer(1, 10);
        assert!(bounds.is_integer);
        assert_eq!(bounds.validate(5.0).unwrap(), 5.0);
        assert_eq!(bounds.validate(5.6).unwrap(), 6.0); // Rounded
    }

    #[test]
    fn test_parameter_bounds_with_floor() {
        let bounds = ParameterBounds::continuous(0.0, 1.0).with_floor(0.5);
        assert_eq!(bounds.effective_min(), 0.5);
        assert!(bounds.validate(0.3).is_err()); // Below floor
        assert!(bounds.validate(0.6).is_ok());
    }

    #[test]
    fn test_parameter_bounds_clamp() {
        let bounds = ParameterBounds::continuous(0.0, 1.0).with_floor(0.3);
        assert_eq!(bounds.clamp(-1.0), 0.3);
        assert_eq!(bounds.clamp(2.0), 1.0);
        assert_eq!(bounds.clamp(0.5), 0.5);
    }

    #[test]
    fn test_constraint_ema_gap() {
        let constraint = Constraint::ema_period_gap("fast", "slow", 10);

        let mut params = HashMap::new();
        params.insert("fast".to_string(), 8.0);
        params.insert("slow".to_string(), 21.0);
        assert!(constraint.validate(&params).is_ok()); // 8 + 10 <= 21

        params.insert("fast".to_string(), 12.0);
        assert!(constraint.validate(&params).is_err()); // 12 + 10 > 21
    }

    #[test]
    fn test_constraint_floor() {
        let constraint = Constraint::floor("min_spread", 0.15);

        let mut params = HashMap::new();
        params.insert("min_spread".to_string(), 0.20);
        assert!(constraint.validate(&params).is_ok());

        params.insert("min_spread".to_string(), 0.10);
        assert!(constraint.validate(&params).is_err());
    }

    #[test]
    fn test_search_space_creation() {
        let space = SearchSpace::ema_strategy();
        assert!(space.num_parameters() > 0);
        assert!(space.get("ema_fast_period").is_some());
        assert!(space.get("ema_slow_period").is_some());
        assert!(space.is_valid());
    }

    #[test]
    fn test_search_space_for_asset() {
        let space = SearchSpace::for_asset("BTC", AssetCategory::Major);
        let bounds = space.get("min_ema_spread_pct").unwrap();
        assert_eq!(bounds.effective_min(), 0.15); // Major coin floor

        let space = SearchSpace::for_asset("DOGE", AssetCategory::Meme);
        let bounds = space.get("min_ema_spread_pct").unwrap();
        assert_eq!(bounds.effective_min(), 0.30); // Meme coin floor
    }

    #[test]
    fn test_search_space_validate() {
        let space = SearchSpace::ema_strategy();

        let mut params = HashMap::new();
        params.insert("ema_fast_period".to_string(), 9.0);
        params.insert("ema_slow_period".to_string(), 28.0);
        params.insert("atr_length".to_string(), 14.0);
        params.insert("atr_multiplier".to_string(), 2.0);
        params.insert("min_trailing_stop_pct".to_string(), 1.0);
        params.insert("min_ema_spread_pct".to_string(), 0.20);
        params.insert("min_profit_pct".to_string(), 0.50);
        params.insert("take_profit_pct".to_string(), 5.0);
        params.insert("require_htf_alignment".to_string(), 1.0);
        params.insert("htf_timeframe_minutes".to_string(), 15.0);
        params.insert("cooldown_bars".to_string(), 5.0);

        assert!(space.validate(&params).is_ok());

        // Violate EMA gap constraint
        params.insert("ema_fast_period".to_string(), 15.0);
        params.insert("ema_slow_period".to_string(), 21.0);
        assert!(space.validate(&params).is_err()); // 15 + 10 > 21
    }

    #[test]
    fn test_sampled_params() {
        let mut values = HashMap::new();
        values.insert("ema_fast".to_string(), 9.0);
        values.insert("require_htf".to_string(), 1.0);

        let params = SampledParams::new(values);
        assert_eq!(params.get("ema_fast"), Some(9.0));
        assert_eq!(params.get_int("ema_fast"), Some(9));
        assert_eq!(params.get_bool("require_htf"), Some(true));
        assert_eq!(params["ema_fast"], 9.0);
    }

    #[test]
    fn test_parameter_bounds_num_values() {
        let bounds = ParameterBounds::integer(1, 10);
        assert_eq!(bounds.num_values(), Some(10));

        let bounds = ParameterBounds::discrete(0.0, 1.0, 0.25);
        assert_eq!(bounds.num_values(), Some(5)); // 0.0, 0.25, 0.5, 0.75, 1.0

        let bounds = ParameterBounds::continuous(0.0, 1.0);
        assert_eq!(bounds.num_values(), None);
    }
}
