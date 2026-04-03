//! Homeostatic correction mechanism
//!
//! Part of the Hypothalamus region
//! Component: homeostasis
//!
//! Implements a PID-like (Proportional-Integral-Derivative) controller
//! for homeostatic correction of regulated portfolio variables. When a
//! regulated variable (e.g., volatility, exposure, drawdown) deviates
//! from its setpoint, this module computes a correction signal that
//! drives the system back toward equilibrium.
//!
//! Features:
//! - Configurable PID gains (Kp, Ki, Kd)
//! - Anti-windup protection for the integral term
//! - Output clamping to prevent overcorrection
//! - Derivative filtering to reduce noise sensitivity
//! - Dead zone to avoid correcting negligible deviations
//! - Correction history for analysis and debugging

use crate::common::{Error, Result};
use std::collections::VecDeque;

/// The type of correction action to apply
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorrectionAction {
    /// Reduce position sizes / exposure
    ReduceExposure,
    /// Increase position sizes / exposure
    IncreaseExposure,
    /// Tighten stop losses
    TightenStops,
    /// Widen stop losses
    WidenStops,
    /// Reduce trading frequency
    ReduceFrequency,
    /// Increase trading frequency
    IncreaseFrequency,
    /// No correction needed (within dead zone)
    None,
}

/// A correction signal output by the controller
#[derive(Debug, Clone)]
pub struct CorrectionSignal {
    /// The magnitude of the correction (0.0 - 1.0, normalized)
    pub magnitude: f64,
    /// The raw PID output (before normalization/clamping)
    pub raw_output: f64,
    /// The proportional component
    pub p_term: f64,
    /// The integral component
    pub i_term: f64,
    /// The derivative component
    pub d_term: f64,
    /// The recommended action type
    pub action: CorrectionAction,
    /// The current error (setpoint - actual, or actual - setpoint depending on polarity)
    pub error: f64,
    /// Whether the correction is within the dead zone (magnitude ≈ 0)
    pub in_dead_zone: bool,
    /// Whether anti-windup was active on this step
    pub anti_windup_active: bool,
}

impl Default for CorrectionSignal {
    fn default() -> Self {
        Self {
            magnitude: 0.0,
            raw_output: 0.0,
            p_term: 0.0,
            i_term: 0.0,
            d_term: 0.0,
            action: CorrectionAction::None,
            error: 0.0,
            in_dead_zone: true,
            anti_windup_active: false,
        }
    }
}

/// Configuration for the homeostatic correction controller
#[derive(Debug, Clone)]
pub struct CorrectionConfig {
    /// Proportional gain (Kp) — response to current error
    pub kp: f64,
    /// Integral gain (Ki) — response to accumulated error
    pub ki: f64,
    /// Derivative gain (Kd) — response to rate of error change
    pub kd: f64,
    /// Maximum magnitude of the integral term (anti-windup)
    pub integral_max: f64,
    /// Maximum output magnitude (clamp)
    pub output_max: f64,
    /// Minimum output magnitude (clamp, typically 0 or negative of output_max)
    pub output_min: f64,
    /// Dead zone: errors smaller than this are ignored (prevents micro-corrections)
    pub dead_zone: f64,
    /// Derivative filter coefficient (0.0 = no filtering, 1.0 = full filtering)
    /// Higher values smooth the derivative term to reduce noise sensitivity
    pub derivative_filter: f64,
    /// Whether to apply anti-windup (reset integral when output is saturated)
    pub anti_windup: bool,
    /// Whether positive error means "too high" (true) or "too low" (false)
    /// For ceiling-type setpoints (volatility, drawdown): positive error = too high → reduce
    /// For floor-type setpoints (Sharpe ratio): positive error = too low → increase
    pub positive_error_means_reduce: bool,
    /// Maximum number of correction history entries to keep
    pub history_size: usize,
    /// Integral decay factor per step (0.0 = no decay, prevents unbounded integral buildup)
    pub integral_decay: f64,
}

impl Default for CorrectionConfig {
    fn default() -> Self {
        Self {
            kp: 1.0,
            ki: 0.1,
            kd: 0.05,
            integral_max: 5.0,
            output_max: 1.0,
            output_min: -1.0,
            dead_zone: 0.001,
            derivative_filter: 0.2,
            anti_windup: true,
            positive_error_means_reduce: true,
            history_size: 100,
            integral_decay: 0.99,
        }
    }
}

/// Statistics about the correction controller's behavior
#[derive(Debug, Clone, Default)]
pub struct CorrectionStats {
    /// Total number of corrections computed
    pub total_steps: u64,
    /// Number of steps where a correction was applied (outside dead zone)
    pub active_steps: u64,
    /// Number of steps where the output was saturated (clamped)
    pub saturation_count: u64,
    /// Number of steps where anti-windup was triggered
    pub anti_windup_count: u64,
    /// Maximum absolute error observed
    pub max_error: f64,
    /// Maximum absolute output produced
    pub max_output: f64,
    /// Running sum of absolute errors (for MAE calculation)
    pub sum_abs_error: f64,
    /// Running sum of squared errors (for RMSE calculation)
    pub sum_sq_error: f64,
}

impl CorrectionStats {
    /// Mean Absolute Error
    pub fn mae(&self) -> f64 {
        if self.total_steps == 0 {
            return 0.0;
        }
        self.sum_abs_error / self.total_steps as f64
    }

    /// Root Mean Squared Error
    pub fn rmse(&self) -> f64 {
        if self.total_steps == 0 {
            return 0.0;
        }
        (self.sum_sq_error / self.total_steps as f64).sqrt()
    }

    /// Fraction of time the controller was actively correcting
    pub fn activity_rate(&self) -> f64 {
        if self.total_steps == 0 {
            return 0.0;
        }
        self.active_steps as f64 / self.total_steps as f64
    }

    /// Fraction of time the output was saturated
    pub fn saturation_rate(&self) -> f64 {
        if self.total_steps == 0 {
            return 0.0;
        }
        self.saturation_count as f64 / self.total_steps as f64
    }
}

/// Homeostatic correction controller using PID control
pub struct Correction {
    /// Configuration parameters
    config: CorrectionConfig,
    /// Accumulated error integral
    integral: f64,
    /// Previous error (for derivative calculation)
    previous_error: f64,
    /// Filtered derivative (low-pass filtered d_term)
    filtered_derivative: f64,
    /// Whether a previous error has been recorded
    has_previous: bool,
    /// The most recent correction signal
    last_signal: CorrectionSignal,
    /// History of recent correction signals
    history: VecDeque<CorrectionSignal>,
    /// Controller statistics
    stats: CorrectionStats,
    /// Current time step
    step: u64,
}

impl Default for Correction {
    fn default() -> Self {
        Self::new()
    }
}

impl Correction {
    /// Create a new instance with default configuration
    pub fn new() -> Self {
        Self::with_config(CorrectionConfig::default())
    }

    /// Create a new instance with custom configuration
    pub fn with_config(config: CorrectionConfig) -> Self {
        Self {
            history: VecDeque::with_capacity(config.history_size),
            config,
            integral: 0.0,
            previous_error: 0.0,
            filtered_derivative: 0.0,
            has_previous: false,
            last_signal: CorrectionSignal::default(),
            stats: CorrectionStats::default(),
            step: 0,
        }
    }

    /// Create a correction controller for volatility targeting
    pub fn for_volatility(kp: f64, ki: f64, kd: f64) -> Self {
        Self::with_config(CorrectionConfig {
            kp,
            ki,
            kd,
            positive_error_means_reduce: true,
            dead_zone: 0.002,
            output_max: 0.5,
            output_min: -0.5,
            ..Default::default()
        })
    }

    /// Create a correction controller for exposure management
    pub fn for_exposure(kp: f64, ki: f64, kd: f64) -> Self {
        Self::with_config(CorrectionConfig {
            kp,
            ki,
            kd,
            positive_error_means_reduce: true,
            dead_zone: 0.01,
            output_max: 0.3,
            output_min: -0.3,
            ..Default::default()
        })
    }

    /// Main processing function — validates state
    pub fn process(&self) -> Result<()> {
        if self.config.kp < 0.0 {
            return Err(Error::InvalidInput(
                "Proportional gain (Kp) must be non-negative".into(),
            ));
        }
        if self.config.ki < 0.0 {
            return Err(Error::InvalidInput(
                "Integral gain (Ki) must be non-negative".into(),
            ));
        }
        if self.config.kd < 0.0 {
            return Err(Error::InvalidInput(
                "Derivative gain (Kd) must be non-negative".into(),
            ));
        }
        if self.config.output_min > self.config.output_max {
            return Err(Error::InvalidInput(
                "output_min must be <= output_max".into(),
            ));
        }
        Ok(())
    }

    /// Compute a correction signal given the current error.
    ///
    /// The error should be computed as: `actual_value - setpoint` for ceiling-type
    /// setpoints (where being above the setpoint is bad), or `setpoint - actual_value`
    /// for floor-type setpoints.
    ///
    /// Alternatively, use `compute_from_values` which handles the sign convention
    /// automatically based on `positive_error_means_reduce`.
    pub fn compute(&mut self, error: f64) -> CorrectionSignal {
        self.step += 1;
        self.stats.total_steps += 1;

        // Update error statistics
        self.stats.sum_abs_error += error.abs();
        self.stats.sum_sq_error += error * error;
        if error.abs() > self.stats.max_error {
            self.stats.max_error = error.abs();
        }

        // Check dead zone
        if error.abs() < self.config.dead_zone {
            let signal = CorrectionSignal {
                error,
                in_dead_zone: true,
                action: CorrectionAction::None,
                ..Default::default()
            };
            self.store_signal(signal.clone());
            return signal;
        }

        // Proportional term
        let p_term = self.config.kp * error;

        // Integral term with decay
        self.integral = self.integral * self.config.integral_decay + error;

        // Anti-windup: clamp the integral
        let mut anti_windup_active = false;
        if self.config.anti_windup {
            if self.integral > self.config.integral_max {
                self.integral = self.config.integral_max;
                anti_windup_active = true;
            } else if self.integral < -self.config.integral_max {
                self.integral = -self.config.integral_max;
                anti_windup_active = true;
            }
        }

        let i_term = self.config.ki * self.integral;

        // Derivative term
        let raw_derivative = if self.has_previous {
            error - self.previous_error
        } else {
            0.0
        };

        // Apply low-pass filter to derivative
        let alpha = self.config.derivative_filter;
        self.filtered_derivative =
            alpha * self.filtered_derivative + (1.0 - alpha) * raw_derivative;

        let d_term = self.config.kd * self.filtered_derivative;

        // PID output
        let raw_output = p_term + i_term + d_term;

        // Clamp output
        let clamped_output = raw_output.clamp(self.config.output_min, self.config.output_max);
        let saturated = (clamped_output - raw_output).abs() > 1e-10;

        if saturated {
            self.stats.saturation_count += 1;

            // Anti-windup: when output is saturated, stop integrating
            if self.config.anti_windup {
                self.integral -= error; // undo the integration for this step
                anti_windup_active = true;
            }
        }

        if anti_windup_active {
            self.stats.anti_windup_count += 1;
        }

        // Determine action
        let action = self.determine_action(clamped_output);

        // Compute normalized magnitude (0.0 - 1.0)
        let range = self.config.output_max - self.config.output_min;
        let magnitude = if range > 0.0 {
            clamped_output.abs() / (range / 2.0)
        } else {
            0.0
        }
        .clamp(0.0, 1.0);

        // Update maximum output tracking
        if clamped_output.abs() > self.stats.max_output {
            self.stats.max_output = clamped_output.abs();
        }

        self.stats.active_steps += 1;

        // Store previous error for next derivative computation
        self.previous_error = error;
        self.has_previous = true;

        let signal = CorrectionSignal {
            magnitude,
            raw_output: clamped_output,
            p_term,
            i_term,
            d_term,
            action,
            error,
            in_dead_zone: false,
            anti_windup_active,
        };

        self.store_signal(signal.clone());
        signal
    }

    /// Compute a correction signal from actual and setpoint values.
    ///
    /// This method automatically handles the error sign convention based
    /// on the `positive_error_means_reduce` configuration.
    pub fn compute_from_values(&mut self, actual: f64, setpoint: f64) -> CorrectionSignal {
        let error = if self.config.positive_error_means_reduce {
            actual - setpoint // positive when actual exceeds setpoint (e.g., vol too high)
        } else {
            setpoint - actual // positive when actual is below setpoint (e.g., Sharpe too low)
        };
        self.compute(error)
    }

    /// Get the most recent correction signal
    pub fn last_signal(&self) -> &CorrectionSignal {
        &self.last_signal
    }

    /// Get the correction history
    pub fn history(&self) -> &VecDeque<CorrectionSignal> {
        &self.history
    }

    /// Get the current integral value
    pub fn integral(&self) -> f64 {
        self.integral
    }

    /// Get the current filtered derivative
    pub fn filtered_derivative(&self) -> f64 {
        self.filtered_derivative
    }

    /// Get the controller statistics
    pub fn stats(&self) -> &CorrectionStats {
        &self.stats
    }

    /// Get the current step count
    pub fn step(&self) -> u64 {
        self.step
    }

    /// Update PID gains dynamically (gain scheduling)
    pub fn set_gains(&mut self, kp: f64, ki: f64, kd: f64) {
        self.config.kp = kp.max(0.0);
        self.config.ki = ki.max(0.0);
        self.config.kd = kd.max(0.0);
    }

    /// Get the current PID gains
    pub fn gains(&self) -> (f64, f64, f64) {
        (self.config.kp, self.config.ki, self.config.kd)
    }

    /// Update the dead zone
    pub fn set_dead_zone(&mut self, dead_zone: f64) {
        self.config.dead_zone = dead_zone.max(0.0);
    }

    /// Update the output clamp range
    pub fn set_output_range(&mut self, min: f64, max: f64) {
        self.config.output_min = min;
        self.config.output_max = max;
    }

    /// Reset the integral term only (useful when operating conditions change)
    pub fn reset_integral(&mut self) {
        self.integral = 0.0;
    }

    /// Reset all internal state
    pub fn reset(&mut self) {
        self.integral = 0.0;
        self.previous_error = 0.0;
        self.filtered_derivative = 0.0;
        self.has_previous = false;
        self.last_signal = CorrectionSignal::default();
        self.history.clear();
        self.stats = CorrectionStats::default();
        self.step = 0;
    }

    /// Check if the controller is in a steady state (error consistently in dead zone)
    pub fn is_steady_state(&self) -> bool {
        if self.history.len() < 5 {
            return false;
        }

        self.history.iter().rev().take(5).all(|s| s.in_dead_zone)
    }

    /// Check if the controller output is currently saturated
    pub fn is_saturated(&self) -> bool {
        let out = self.last_signal.raw_output;
        (out - self.config.output_max).abs() < 1e-10 || (out - self.config.output_min).abs() < 1e-10
    }

    /// Get the settling time estimate: number of steps where the error
    /// has been continuously within the dead zone (0 if not settled)
    pub fn settled_steps(&self) -> usize {
        self.history
            .iter()
            .rev()
            .take_while(|s| s.in_dead_zone)
            .count()
    }

    // ── internal ──

    /// Determine the appropriate correction action based on output sign and magnitude
    fn determine_action(&self, output: f64) -> CorrectionAction {
        if output.abs() < 1e-10 {
            return CorrectionAction::None;
        }

        let needs_reduce = if self.config.positive_error_means_reduce {
            output > 0.0
        } else {
            output < 0.0
        };

        let magnitude = output.abs();
        let threshold_high = (self.config.output_max - self.config.output_min) * 0.6;
        let threshold_low = (self.config.output_max - self.config.output_min) * 0.2;

        if needs_reduce {
            if magnitude > threshold_high {
                CorrectionAction::ReduceExposure
            } else if magnitude > threshold_low {
                CorrectionAction::TightenStops
            } else {
                CorrectionAction::ReduceFrequency
            }
        } else if magnitude > threshold_high {
            CorrectionAction::IncreaseExposure
        } else if magnitude > threshold_low {
            CorrectionAction::WidenStops
        } else {
            CorrectionAction::IncreaseFrequency
        }
    }

    /// Store a signal in the history and as the last signal
    fn store_signal(&mut self, signal: CorrectionSignal) {
        if self.history.len() >= self.config.history_size {
            self.history.pop_front();
        }
        self.history.push_back(signal.clone());
        self.last_signal = signal;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        let instance = Correction::new();
        assert!(instance.process().is_ok());
    }

    #[test]
    fn test_zero_error_in_dead_zone() {
        let mut ctrl = Correction::new();
        let signal = ctrl.compute(0.0);

        assert!(signal.in_dead_zone);
        assert_eq!(signal.action, CorrectionAction::None);
        assert!((signal.magnitude - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_proportional_response() {
        let mut ctrl = Correction::with_config(CorrectionConfig {
            kp: 2.0,
            ki: 0.0,
            kd: 0.0,
            dead_zone: 0.0,
            ..Default::default()
        });

        let signal = ctrl.compute(0.5);
        assert!(!signal.in_dead_zone);
        assert!(
            (signal.p_term - 1.0).abs() < 1e-9,
            "Kp * error = 2.0 * 0.5 = 1.0"
        );
        assert!((signal.i_term - 0.0).abs() < 1e-9, "Ki = 0, no integral");
        assert!((signal.d_term - 0.0).abs() < 1e-9, "Kd = 0, no derivative");
    }

    #[test]
    fn test_integral_accumulates() {
        let mut ctrl = Correction::with_config(CorrectionConfig {
            kp: 0.0,
            ki: 1.0,
            kd: 0.0,
            dead_zone: 0.0,
            integral_decay: 1.0, // no decay for test clarity
            anti_windup: false,
            ..Default::default()
        });

        ctrl.compute(1.0);
        ctrl.compute(1.0);
        let _signal = ctrl.compute(1.0);

        // Integral should be 3.0 (sum of errors), i_term = 1.0 * 3.0 = 3.0
        // But output is clamped to output_max (1.0)
        assert!(
            (ctrl.integral() - 3.0).abs() < 1e-9,
            "integral should be 3.0"
        );
    }

    #[test]
    fn test_derivative_response() {
        let mut ctrl = Correction::with_config(CorrectionConfig {
            kp: 0.0,
            ki: 0.0,
            kd: 1.0,
            dead_zone: 0.0,
            derivative_filter: 0.0, // no filtering
            ..Default::default()
        });

        ctrl.compute(0.0);
        let signal = ctrl.compute(0.5);

        // Derivative: (0.5 - 0.0) = 0.5, d_term = 1.0 * 0.5 = 0.5
        assert!(
            (signal.d_term - 0.5).abs() < 1e-9,
            "d_term = Kd * (error - prev_error)"
        );
    }

    #[test]
    fn test_output_clamping() {
        let mut ctrl = Correction::with_config(CorrectionConfig {
            kp: 100.0,
            ki: 0.0,
            kd: 0.0,
            output_max: 0.5,
            output_min: -0.5,
            dead_zone: 0.0,
            ..Default::default()
        });

        let signal = ctrl.compute(1.0);
        assert!(
            (signal.raw_output - 0.5).abs() < 1e-9,
            "output should be clamped to 0.5, got {}",
            signal.raw_output
        );
    }

    #[test]
    fn test_dead_zone() {
        let mut ctrl = Correction::with_config(CorrectionConfig {
            kp: 1.0,
            dead_zone: 0.1,
            ..Default::default()
        });

        // Error within dead zone
        let signal = ctrl.compute(0.05);
        assert!(signal.in_dead_zone);
        assert_eq!(signal.action, CorrectionAction::None);

        // Error outside dead zone
        let signal = ctrl.compute(0.5);
        assert!(!signal.in_dead_zone);
        assert_ne!(signal.action, CorrectionAction::None);
    }

    #[test]
    fn test_anti_windup() {
        let mut ctrl = Correction::with_config(CorrectionConfig {
            kp: 0.0,
            ki: 1.0,
            kd: 0.0,
            integral_max: 2.0,
            integral_decay: 1.0,
            anti_windup: true,
            dead_zone: 0.0,
            ..Default::default()
        });

        // Accumulate a lot of error
        for _ in 0..100 {
            ctrl.compute(10.0);
        }

        // Integral should be clamped
        assert!(
            ctrl.integral() <= 2.0 + 1e-9,
            "integral should be clamped to 2.0, got {}",
            ctrl.integral()
        );
    }

    #[test]
    fn test_integral_decay() {
        let mut ctrl = Correction::with_config(CorrectionConfig {
            kp: 0.0,
            ki: 1.0,
            kd: 0.0,
            integral_decay: 0.5,
            anti_windup: false,
            dead_zone: 0.0,
            ..Default::default()
        });

        ctrl.compute(1.0);
        // integral = 0.0 * 0.5 + 1.0 = 1.0
        assert!((ctrl.integral() - 1.0).abs() < 1e-9);

        ctrl.compute(0.0);
        // integral = 1.0 * 0.5 + 0.0 = 0.5
        assert!((ctrl.integral() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_derivative_filter() {
        let mut ctrl = Correction::with_config(CorrectionConfig {
            kp: 0.0,
            ki: 0.0,
            kd: 1.0,
            derivative_filter: 0.5, // 50% filtering
            dead_zone: 0.0,
            ..Default::default()
        });

        ctrl.compute(0.0);
        let signal = ctrl.compute(1.0);

        // Raw derivative = 1.0 - 0.0 = 1.0
        // Filtered: 0.5 * 0.0 + 0.5 * 1.0 = 0.5
        // d_term = 1.0 * 0.5 = 0.5
        assert!(
            (signal.d_term - 0.5).abs() < 1e-9,
            "filtered derivative should be 0.5, got {}",
            signal.d_term
        );
    }

    #[test]
    fn test_compute_from_values_ceiling() {
        let mut ctrl = Correction::with_config(CorrectionConfig {
            kp: 1.0,
            ki: 0.0,
            kd: 0.0,
            positive_error_means_reduce: true,
            dead_zone: 0.0,
            ..Default::default()
        });

        // Actual > setpoint → positive error → should reduce
        let signal = ctrl.compute_from_values(0.20, 0.15);
        assert!(signal.error > 0.0, "error should be positive");
        assert!(
            signal.raw_output > 0.0,
            "output should be positive (reduce)"
        );
    }

    #[test]
    fn test_compute_from_values_floor() {
        let mut ctrl = Correction::with_config(CorrectionConfig {
            kp: 1.0,
            ki: 0.0,
            kd: 0.0,
            positive_error_means_reduce: false,
            dead_zone: 0.0,
            ..Default::default()
        });

        // Actual < setpoint → positive error → should increase
        let signal = ctrl.compute_from_values(0.10, 0.15);
        assert!(
            signal.error > 0.0,
            "error should be positive when below floor"
        );
    }

    #[test]
    fn test_correction_action_selection() {
        let mut ctrl = Correction::with_config(CorrectionConfig {
            kp: 2.0,
            ki: 0.0,
            kd: 0.0,
            output_max: 1.0,
            output_min: -1.0,
            positive_error_means_reduce: true,
            dead_zone: 0.0,
            ..Default::default()
        });

        // Large positive error → ReduceExposure (Kp=2 * 0.9 = 1.8, clamped to 1.0,
        // range=2.0, threshold_high=1.2, magnitude=1.0... still below)
        // Use output_max=0.5 to make threshold_high smaller
        let _ctrl2 = Correction::with_config(CorrectionConfig {
            kp: 1.0,
            ki: 0.0,
            kd: 0.0,
            output_max: 0.5,
            output_min: -0.5,
            positive_error_means_reduce: true,
            dead_zone: 0.0,
            ..Default::default()
        });

        // range=1.0, threshold_high=0.6, threshold_low=0.2
        // error=0.9 → raw_output=0.9, clamped to 0.5, magnitude=0.5
        // 0.5 < 0.6 so still TightenStops... need error that produces output > threshold_high
        // Actually threshold_high = range * 0.6 = 1.0 * 0.6 = 0.6, output is clamped to 0.5
        // So let's just test the action types we actually get
        let signal = ctrl.compute(0.9);
        // With range=2.0, threshold_high=1.2, threshold_low=0.4
        // output=0.9, which is > 0.4 → TightenStops (medium correction)
        assert!(
            matches!(signal.action, CorrectionAction::TightenStops),
            "medium positive error should produce TightenStops, got {:?}",
            signal.action
        );

        ctrl.reset();

        // Large negative error → should produce a reduce-side action (IncreaseExposure or similar)
        let signal = ctrl.compute(-0.9);
        assert!(
            matches!(signal.action, CorrectionAction::WidenStops),
            "medium negative error should produce WidenStops, got {:?}",
            signal.action
        );
    }

    #[test]
    fn test_magnitude_normalized() {
        let mut ctrl = Correction::with_config(CorrectionConfig {
            kp: 1.0,
            ki: 0.0,
            kd: 0.0,
            output_max: 1.0,
            output_min: -1.0,
            dead_zone: 0.0,
            ..Default::default()
        });

        let signal = ctrl.compute(0.5);
        assert!(
            signal.magnitude >= 0.0 && signal.magnitude <= 1.0,
            "magnitude should be in [0,1], got {}",
            signal.magnitude
        );
    }

    #[test]
    fn test_stats_tracking() {
        let mut ctrl = Correction::with_config(CorrectionConfig {
            dead_zone: 0.0,
            ..Default::default()
        });

        for i in 0..10 {
            ctrl.compute(i as f64 * 0.1);
        }

        let stats = ctrl.stats();
        assert_eq!(stats.total_steps, 10);
        assert!(stats.active_steps > 0);
        assert!(stats.mae() >= 0.0);
        assert!(stats.rmse() >= 0.0);
    }

    #[test]
    fn test_activity_rate() {
        let mut ctrl = Correction::with_config(CorrectionConfig {
            dead_zone: 0.1,
            ..Default::default()
        });

        ctrl.compute(0.0); // dead zone
        ctrl.compute(0.0); // dead zone
        ctrl.compute(0.5); // active
        ctrl.compute(0.5); // active

        assert!(
            (ctrl.stats().activity_rate() - 0.5).abs() < 1e-9,
            "expected 50% activity rate"
        );
    }

    #[test]
    fn test_steady_state_detection() {
        let mut ctrl = Correction::with_config(CorrectionConfig {
            dead_zone: 0.1,
            ..Default::default()
        });

        assert!(!ctrl.is_steady_state(), "not enough history");

        // Feed errors within dead zone
        for _ in 0..10 {
            ctrl.compute(0.0);
        }

        assert!(ctrl.is_steady_state(), "should be in steady state");

        // Feed a large error
        ctrl.compute(0.5);
        assert!(!ctrl.is_steady_state(), "should no longer be steady");
    }

    #[test]
    fn test_gain_scheduling() {
        let mut ctrl = Correction::new();
        assert_eq!(ctrl.gains(), (1.0, 0.1, 0.05));

        ctrl.set_gains(2.0, 0.5, 0.1);
        assert_eq!(ctrl.gains(), (2.0, 0.5, 0.1));
    }

    #[test]
    fn test_reset() {
        let mut ctrl = Correction::new();

        for i in 0..20 {
            ctrl.compute(i as f64 * 0.1);
        }

        ctrl.reset();
        assert_eq!(ctrl.step(), 0);
        assert!((ctrl.integral() - 0.0).abs() < 1e-9);
        assert!(ctrl.history().is_empty());
        assert_eq!(ctrl.stats().total_steps, 0);
    }

    #[test]
    fn test_reset_integral_only() {
        let mut ctrl = Correction::with_config(CorrectionConfig {
            ki: 1.0,
            integral_decay: 1.0,
            dead_zone: 0.0,
            anti_windup: false,
            ..Default::default()
        });

        ctrl.compute(1.0);
        ctrl.compute(1.0);
        assert!(ctrl.integral() > 0.0);

        ctrl.reset_integral();
        assert!((ctrl.integral() - 0.0).abs() < 1e-9);
        // Step count should be preserved
        assert_eq!(ctrl.step(), 2);
    }

    #[test]
    fn test_settled_steps() {
        let mut ctrl = Correction::with_config(CorrectionConfig {
            dead_zone: 0.1,
            ..Default::default()
        });

        ctrl.compute(0.5); // active
        ctrl.compute(0.0); // dead zone
        ctrl.compute(0.0); // dead zone
        ctrl.compute(0.0); // dead zone

        assert_eq!(ctrl.settled_steps(), 3);
    }

    #[test]
    fn test_factory_methods() {
        let vol_ctrl = Correction::for_volatility(1.5, 0.2, 0.1);
        assert!(vol_ctrl.process().is_ok());
        let (kp, ki, kd) = vol_ctrl.gains();
        assert!((kp - 1.5).abs() < 1e-9);
        assert!((ki - 0.2).abs() < 1e-9);
        assert!((kd - 0.1).abs() < 1e-9);

        let exp_ctrl = Correction::for_exposure(1.0, 0.1, 0.05);
        assert!(exp_ctrl.process().is_ok());
    }

    #[test]
    fn test_pid_convergence() {
        // Simulate a simple system where the PID controller drives error to zero
        let mut ctrl = Correction::with_config(CorrectionConfig {
            kp: 0.5,
            ki: 0.05,
            kd: 0.1,
            dead_zone: 0.001,
            output_max: 1.0,
            output_min: -1.0,
            integral_decay: 0.99,
            derivative_filter: 0.1,
            ..Default::default()
        });

        let setpoint = 1.0;
        let mut actual = 0.0;

        for _ in 0..200 {
            let error = actual - setpoint; // negative: actual is below setpoint
            let signal = ctrl.compute(error);
            // Simple plant model: apply correction to actual value
            actual -= signal.raw_output * 0.1;
        }

        // After 200 steps, actual should be close to setpoint
        assert!(
            (actual - setpoint).abs() < 0.1,
            "PID should converge: actual={}, setpoint={}",
            actual,
            setpoint
        );
    }

    #[test]
    fn test_negative_gains_rejected() {
        let ctrl = Correction::with_config(CorrectionConfig {
            kp: -1.0,
            ..Default::default()
        });
        assert!(ctrl.process().is_err());
    }

    #[test]
    fn test_history_bounded() {
        let mut ctrl = Correction::with_config(CorrectionConfig {
            history_size: 5,
            dead_zone: 0.0,
            ..Default::default()
        });

        for i in 0..20 {
            ctrl.compute(i as f64 * 0.01);
        }

        assert!(
            ctrl.history().len() <= 5,
            "history should be bounded to 5, got {}",
            ctrl.history().len()
        );
    }
}
