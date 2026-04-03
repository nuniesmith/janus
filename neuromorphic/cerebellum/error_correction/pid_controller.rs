//! PID Controller for Execution Error Correction
//!
//! Part of the Cerebellum region - provides real-time feedback control
//! for execution parameters like order sizing, timing, and price adjustments.
//!
//! The PID (Proportional-Integral-Derivative) controller is used to:
//! - Minimize execution slippage
//! - Maintain target fill rates
//! - Adjust order aggressiveness based on market conditions
//!
//! # Example
//!
//! ```rust,ignore
//! use cerebellum::error_correction::PidController;
//!
//! let mut controller = PidController::new(PidConfig {
//!     kp: 0.5,  // Proportional gain
//!     ki: 0.1,  // Integral gain
//!     kd: 0.05, // Derivative gain
//!     ..Default::default()
//! });
//!
//! // Update with execution error (target - actual)
//! let correction = controller.update(error, dt);
//! ```

use crate::common::Result;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::Instant;

/// PID controller configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PidConfig {
    /// Proportional gain (Kp)
    /// Higher values = stronger immediate response to error
    pub kp: f64,

    /// Integral gain (Ki)
    /// Higher values = stronger response to accumulated error
    pub ki: f64,

    /// Derivative gain (Kd)
    /// Higher values = stronger response to rate of change
    pub kd: f64,

    /// Output minimum limit
    pub output_min: f64,

    /// Output maximum limit
    pub output_max: f64,

    /// Integral windup limit (prevents runaway integration)
    pub integral_limit: f64,

    /// Derivative filter coefficient (0-1, higher = more filtering)
    pub derivative_filter: f64,

    /// Dead band - ignore errors smaller than this
    pub dead_band: f64,

    /// Sample time for discrete PID (seconds)
    pub sample_time: f64,

    /// Enable anti-windup mechanism
    pub anti_windup: bool,

    /// Setpoint weighting for proportional term (0-1)
    pub setpoint_weight_p: f64,

    /// Setpoint weighting for derivative term (0-1)
    pub setpoint_weight_d: f64,
}

impl Default for PidConfig {
    fn default() -> Self {
        Self {
            kp: 1.0,
            ki: 0.1,
            kd: 0.05,
            output_min: -1.0,
            output_max: 1.0,
            integral_limit: 10.0,
            derivative_filter: 0.1,
            dead_band: 0.001,
            sample_time: 0.1,
            anti_windup: true,
            setpoint_weight_p: 1.0,
            setpoint_weight_d: 0.0,
        }
    }
}

/// PID controller state
#[derive(Debug, Clone)]
pub struct PidState {
    /// Accumulated integral term
    pub integral: f64,

    /// Previous error (for derivative calculation)
    pub prev_error: f64,

    /// Previous derivative (for filtering)
    pub prev_derivative: f64,

    /// Previous output (for anti-windup)
    pub prev_output: f64,

    /// Previous setpoint (for derivative-on-measurement)
    pub prev_setpoint: f64,

    /// Previous process variable
    pub prev_pv: f64,

    /// Last update time
    pub last_update: Option<Instant>,

    /// Is this the first update?
    pub first_update: bool,
}

impl Default for PidState {
    fn default() -> Self {
        Self {
            integral: 0.0,
            prev_error: 0.0,
            prev_derivative: 0.0,
            prev_output: 0.0,
            prev_setpoint: 0.0,
            prev_pv: 0.0,
            last_update: None,
            first_update: true,
        }
    }
}

/// PID controller output details
#[derive(Debug, Clone, Serialize)]
pub struct PidOutput {
    /// Total control output
    pub output: f64,

    /// Proportional component
    pub p_term: f64,

    /// Integral component
    pub i_term: f64,

    /// Derivative component
    pub d_term: f64,

    /// Current error
    pub error: f64,

    /// Whether output was saturated
    pub saturated: bool,
}

/// PID control for execution error correction
#[derive(Debug, Clone)]
pub struct PidController {
    /// Controller configuration
    config: PidConfig,

    /// Controller state
    state: PidState,

    /// History of errors for analysis
    error_history: VecDeque<(Instant, f64)>,

    /// Maximum history size
    max_history: usize,
}

impl Default for PidController {
    fn default() -> Self {
        Self::new()
    }
}

impl PidController {
    /// Create a new PID controller with default configuration
    pub fn new() -> Self {
        Self::with_config(PidConfig::default())
    }

    /// Create a new PID controller with custom configuration
    pub fn with_config(config: PidConfig) -> Self {
        Self {
            config,
            state: PidState::default(),
            error_history: VecDeque::new(),
            max_history: 1000,
        }
    }

    /// Create a PID controller with simple P-I-D gains
    pub fn with_gains(kp: f64, ki: f64, kd: f64) -> Self {
        Self::with_config(PidConfig {
            kp,
            ki,
            kd,
            ..Default::default()
        })
    }

    /// Update the controller with a new error value
    ///
    /// # Arguments
    /// * `error` - The current error (setpoint - process_variable)
    /// * `dt` - Time step in seconds (if None, uses elapsed time since last update)
    ///
    /// # Returns
    /// * The control output and detailed components
    pub fn update(&mut self, error: f64, dt: Option<f64>) -> PidOutput {
        let now = Instant::now();

        // Calculate time step
        let dt = dt.unwrap_or_else(|| {
            self.state
                .last_update
                .map(|t| now.duration_since(t).as_secs_f64())
                .unwrap_or(self.config.sample_time)
        });

        // Apply dead band
        let error = if error.abs() < self.config.dead_band {
            0.0
        } else {
            error
        };

        // Calculate proportional term
        let p_term = self.config.kp * error;

        // Calculate integral term with anti-windup
        let mut i_term = self.state.integral;
        if dt > 0.0 {
            i_term += self.config.ki * error * dt;

            // Apply integral windup limit
            i_term = i_term.clamp(-self.config.integral_limit, self.config.integral_limit);
        }

        // Calculate derivative term with filtering
        let d_term = if self.state.first_update || dt <= 0.0 {
            0.0
        } else {
            let raw_derivative = (error - self.state.prev_error) / dt;

            // Apply low-pass filter to derivative
            let alpha = self.config.derivative_filter;
            let filtered_derivative =
                alpha * self.state.prev_derivative + (1.0 - alpha) * raw_derivative;

            self.config.kd * filtered_derivative
        };

        // Calculate raw output
        let raw_output = p_term + i_term + d_term;

        // Apply output limits
        let output = raw_output.clamp(self.config.output_min, self.config.output_max);
        let saturated = (raw_output - output).abs() > 1e-10;

        // Anti-windup: if output is saturated, don't accumulate more integral
        if self.config.anti_windup && saturated {
            // Back-calculate integral to prevent windup
            if (raw_output > self.config.output_max && error > 0.0)
                || (raw_output < self.config.output_min && error < 0.0)
            {
                // Don't update integral when saturating in the error direction
            } else {
                self.state.integral = i_term;
            }
        } else {
            self.state.integral = i_term;
        }

        // Update state (compute derivative BEFORE overwriting prev_error)
        self.state.prev_derivative = if dt > 0.0 {
            (error - self.state.prev_error) / dt
        } else {
            self.state.prev_derivative
        };
        self.state.prev_error = error;
        self.state.prev_output = output;
        self.state.last_update = Some(now);
        self.state.first_update = false;

        // Record error history
        self.error_history.push_back((now, error));
        while self.error_history.len() > self.max_history {
            self.error_history.pop_front();
        }

        PidOutput {
            output,
            p_term,
            i_term: self.state.integral,
            d_term,
            error,
            saturated,
        }
    }

    /// Update with setpoint and process variable separately
    ///
    /// This allows for setpoint weighting and derivative-on-measurement
    pub fn update_with_pv(
        &mut self,
        setpoint: f64,
        process_variable: f64,
        dt: Option<f64>,
    ) -> PidOutput {
        let now = Instant::now();

        let dt = dt.unwrap_or_else(|| {
            self.state
                .last_update
                .map(|t| now.duration_since(t).as_secs_f64())
                .unwrap_or(self.config.sample_time)
        });

        let error = setpoint - process_variable;

        // Apply dead band
        let error = if error.abs() < self.config.dead_band {
            0.0
        } else {
            error
        };

        // Proportional term with setpoint weighting
        let p_error = self.config.setpoint_weight_p * setpoint - process_variable;
        let p_term = self.config.kp * p_error;

        // Integral term
        let mut i_term = self.state.integral;
        if dt > 0.0 {
            i_term += self.config.ki * error * dt;
            i_term = i_term.clamp(-self.config.integral_limit, self.config.integral_limit);
        }

        // Derivative term on measurement (not setpoint) to avoid derivative kick
        let d_term = if self.state.first_update || dt <= 0.0 {
            0.0
        } else {
            // Derivative on measurement (negative because we want d(error)/dt)
            let d_input = if self.config.setpoint_weight_d == 0.0 {
                // Pure derivative on measurement
                -(process_variable - self.state.prev_pv) / dt
            } else {
                // Weighted combination
                let sp_contrib =
                    self.config.setpoint_weight_d * (setpoint - self.state.prev_setpoint);
                let pv_contrib = -(1.0 - self.config.setpoint_weight_d)
                    * (process_variable - self.state.prev_pv);
                (sp_contrib + pv_contrib) / dt
            };

            let alpha = self.config.derivative_filter;
            let filtered = alpha * self.state.prev_derivative + (1.0 - alpha) * d_input;
            self.config.kd * filtered
        };

        let raw_output = p_term + i_term + d_term;
        let output = raw_output.clamp(self.config.output_min, self.config.output_max);
        let saturated = (raw_output - output).abs() > 1e-10;

        // Anti-windup
        if self.config.anti_windup && saturated {
            if !((raw_output > self.config.output_max && error > 0.0)
                || (raw_output < self.config.output_min && error < 0.0))
            {
                self.state.integral = i_term;
            }
        } else {
            self.state.integral = i_term;
        }

        // Update state
        self.state.prev_error = error;
        self.state.prev_setpoint = setpoint;
        self.state.prev_pv = process_variable;
        self.state.prev_derivative = if dt > 0.0 {
            -(process_variable - self.state.prev_pv) / dt
        } else {
            self.state.prev_derivative
        };
        self.state.prev_output = output;
        self.state.last_update = Some(now);
        self.state.first_update = false;

        self.error_history.push_back((now, error));
        while self.error_history.len() > self.max_history {
            self.error_history.pop_front();
        }

        PidOutput {
            output,
            p_term,
            i_term: self.state.integral,
            d_term,
            error,
            saturated,
        }
    }

    /// Reset the controller state
    pub fn reset(&mut self) {
        self.state = PidState::default();
        self.error_history.clear();
    }

    /// Set new gains dynamically
    pub fn set_gains(&mut self, kp: f64, ki: f64, kd: f64) {
        self.config.kp = kp;
        self.config.ki = ki;
        self.config.kd = kd;
    }

    /// Set output limits
    pub fn set_limits(&mut self, min: f64, max: f64) {
        self.config.output_min = min;
        self.config.output_max = max;
    }

    /// Get current integral value
    pub fn get_integral(&self) -> f64 {
        self.state.integral
    }

    /// Manually set integral (for bumpless transfer)
    pub fn set_integral(&mut self, integral: f64) {
        self.state.integral =
            integral.clamp(-self.config.integral_limit, self.config.integral_limit);
    }

    /// Get error statistics
    pub fn get_error_stats(&self) -> ErrorStats {
        if self.error_history.is_empty() {
            return ErrorStats::default();
        }

        let errors: Vec<f64> = self.error_history.iter().map(|(_, e)| *e).collect();
        let n = errors.len() as f64;

        let mean = errors.iter().sum::<f64>() / n;
        let variance = errors.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();

        let min = errors.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = errors.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Mean absolute error
        let mae = errors.iter().map(|e| e.abs()).sum::<f64>() / n;

        // Root mean square error
        let rmse = (errors.iter().map(|e| e.powi(2)).sum::<f64>() / n).sqrt();

        ErrorStats {
            mean,
            std_dev,
            min,
            max,
            mae,
            rmse,
            sample_count: errors.len(),
        }
    }

    /// Auto-tune PID gains using Ziegler-Nichols method
    ///
    /// This requires finding the ultimate gain (Ku) and period (Tu)
    /// through experimentation or relay feedback
    pub fn auto_tune_ziegler_nichols(&mut self, ku: f64, tu: f64, controller_type: ZNType) {
        let (kp, ki, kd) = match controller_type {
            ZNType::P => (0.5 * ku, 0.0, 0.0),
            ZNType::PI => (0.45 * ku, 1.2 * ku / tu, 0.0),
            ZNType::PID => (0.6 * ku, 2.0 * ku / tu, ku * tu / 8.0),
            ZNType::PIDNoOvershoot => (0.2 * ku, 0.8 * ku / tu, ku * tu / 15.0),
            ZNType::PessenIntegral => (0.7 * ku, 2.5 * ku / tu, 0.15 * ku * tu),
            ZNType::SomeOvershoot => (0.33 * ku, 1.32 * ku / tu, 0.083 * ku * tu),
        };

        self.set_gains(kp, ki, kd);
    }

    /// Get configuration
    pub fn config(&self) -> &PidConfig {
        &self.config
    }

    /// Get mutable configuration
    pub fn config_mut(&mut self) -> &mut PidConfig {
        &mut self.config
    }

    /// Main processing function (for compatibility)
    pub fn process(&self) -> Result<()> {
        Ok(())
    }
}

/// Error statistics
#[derive(Debug, Clone, Default, Serialize)]
pub struct ErrorStats {
    /// Mean error
    pub mean: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Minimum error
    pub min: f64,
    /// Maximum error
    pub max: f64,
    /// Mean absolute error
    pub mae: f64,
    /// Root mean square error
    pub rmse: f64,
    /// Number of samples
    pub sample_count: usize,
}

/// Ziegler-Nichols tuning types
#[derive(Debug, Clone, Copy)]
pub enum ZNType {
    /// P controller only
    P,
    /// PI controller
    PI,
    /// Classic PID
    PID,
    /// PID with no overshoot
    PIDNoOvershoot,
    /// Pessen Integral Rule
    PessenIntegral,
    /// Some overshoot rule
    SomeOvershoot,
}

/// Cascaded PID controller for multi-loop control
#[derive(Debug)]
pub struct CascadedPid {
    /// Outer (primary) controller
    outer: PidController,
    /// Inner (secondary) controller
    inner: PidController,
}

impl CascadedPid {
    /// Create a new cascaded PID controller
    pub fn new(outer_config: PidConfig, inner_config: PidConfig) -> Self {
        Self {
            outer: PidController::with_config(outer_config),
            inner: PidController::with_config(inner_config),
        }
    }

    /// Update the cascaded controller
    ///
    /// # Arguments
    /// * `primary_setpoint` - Setpoint for outer loop
    /// * `primary_pv` - Process variable for outer loop
    /// * `secondary_pv` - Process variable for inner loop
    /// * `dt` - Time step
    pub fn update(
        &mut self,
        primary_setpoint: f64,
        primary_pv: f64,
        secondary_pv: f64,
        dt: Option<f64>,
    ) -> (PidOutput, PidOutput) {
        // Outer loop generates setpoint for inner loop
        let outer_output = self.outer.update_with_pv(primary_setpoint, primary_pv, dt);

        // Inner loop tracks outer loop output
        let inner_output = self
            .inner
            .update_with_pv(outer_output.output, secondary_pv, dt);

        (outer_output, inner_output)
    }

    /// Reset both controllers
    pub fn reset(&mut self) {
        self.outer.reset();
        self.inner.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        let instance = PidController::new();
        assert!(instance.process().is_ok());
    }

    #[test]
    fn test_proportional_response() {
        let mut pid = PidController::with_gains(1.0, 0.0, 0.0);

        let output = pid.update(1.0, Some(0.1));
        assert!((output.output - 1.0).abs() < 0.01);
        assert!((output.p_term - 1.0).abs() < 0.01);
        assert!((output.i_term).abs() < 0.01);
        assert!((output.d_term).abs() < 0.01);
    }

    #[test]
    fn test_integral_accumulation() {
        let mut pid = PidController::with_gains(0.0, 1.0, 0.0);

        // Apply constant error over time
        let _ = pid.update(1.0, Some(0.1));
        let output2 = pid.update(1.0, Some(0.1));
        let output3 = pid.update(1.0, Some(0.1));

        // Integral should accumulate
        assert!(output3.i_term > output2.i_term);
    }

    #[test]
    fn test_derivative_response() {
        let mut pid = PidController::with_gains(0.0, 0.0, 1.0);

        // First update - no derivative yet
        let _ = pid.update(0.0, Some(0.1));

        // Sudden change in error
        let output = pid.update(1.0, Some(0.1));

        // Derivative should respond to change
        assert!(output.d_term.abs() > 0.0);
    }

    #[test]
    fn test_output_saturation() {
        let mut pid = PidController::with_config(PidConfig {
            kp: 10.0,
            ki: 0.0,
            kd: 0.0,
            output_min: -1.0,
            output_max: 1.0,
            ..Default::default()
        });

        let output = pid.update(5.0, Some(0.1));
        assert!((output.output - 1.0).abs() < 0.01);
        assert!(output.saturated);
    }

    #[test]
    fn test_dead_band() {
        let mut pid = PidController::with_config(PidConfig {
            kp: 1.0,
            ki: 0.0,
            kd: 0.0,
            dead_band: 0.1,
            ..Default::default()
        });

        let output = pid.update(0.05, Some(0.1));
        assert!((output.error).abs() < 0.01);
        assert!((output.output).abs() < 0.01);
    }

    #[test]
    fn test_reset() {
        // Use small kp so that p_term alone doesn't saturate the output
        // (default output_max = 1.0), allowing integral to accumulate.
        let mut pid = PidController::with_config(PidConfig {
            kp: 0.1,
            ki: 1.0,
            kd: 0.0,
            ..Default::default()
        });

        // Accumulate some state with explicit dt to avoid timing issues
        for _ in 0..5 {
            let _ = pid.update(0.5, Some(0.1));
        }

        let integral_before = pid.get_integral();
        assert!(
            integral_before.abs() > 0.0,
            "integral should be non-zero after updates, got {}",
            integral_before
        );

        // Reset
        pid.reset();

        assert!(
            pid.get_integral().abs() < 1e-12,
            "integral should be zero after reset, got {}",
            pid.get_integral()
        );
    }

    #[test]
    fn test_anti_windup() {
        let mut pid = PidController::with_config(PidConfig {
            kp: 0.0,
            ki: 10.0,
            kd: 0.0,
            output_min: -1.0,
            output_max: 1.0,
            integral_limit: 100.0,
            anti_windup: true,
            ..Default::default()
        });

        // Apply large error repeatedly
        for _ in 0..100 {
            let _ = pid.update(10.0, Some(0.1));
        }

        // Integral should be limited
        let integral = pid.get_integral();
        assert!(integral <= 100.0);
    }

    #[test]
    fn test_error_stats() {
        let mut pid = PidController::new();

        // Generate some errors
        pid.update(1.0, Some(0.1));
        pid.update(-1.0, Some(0.1));
        pid.update(0.5, Some(0.1));
        pid.update(-0.5, Some(0.1));

        let stats = pid.get_error_stats();
        assert_eq!(stats.sample_count, 4);
        assert!(stats.std_dev > 0.0);
    }

    #[test]
    fn test_ziegler_nichols_tuning() {
        let mut pid = PidController::new();

        // Apply Ziegler-Nichols tuning
        pid.auto_tune_ziegler_nichols(2.0, 1.0, ZNType::PID);

        let config = pid.config();
        assert!((config.kp - 1.2).abs() < 0.01); // 0.6 * Ku
        assert!((config.ki - 4.0).abs() < 0.01); // 2.0 * Ku / Tu
        assert!((config.kd - 0.25).abs() < 0.01); // Ku * Tu / 8
    }

    #[test]
    fn test_cascaded_pid() {
        let outer_config = PidConfig {
            kp: 1.0,
            ki: 0.1,
            kd: 0.0,
            ..Default::default()
        };
        let inner_config = PidConfig {
            kp: 2.0,
            ki: 0.2,
            kd: 0.0,
            ..Default::default()
        };

        let mut cascade = CascadedPid::new(outer_config, inner_config);

        let (outer, inner) = cascade.update(1.0, 0.0, 0.0, Some(0.1));

        // Outer loop should generate a setpoint
        assert!(outer.output.abs() > 0.0);
        // Inner loop should track it
        assert!(inner.output.abs() > 0.0);
    }
}
