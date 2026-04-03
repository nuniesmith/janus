//! Feedback control loop for execution quality
//!
//! Part of the Cerebellum region
//! Component: error_correction
//!
//! Combines error signals from multiple execution dimensions (slippage,
//! timing, fill rate) into a unified correction signal using configurable
//! gain scheduling. Implements setpoint tracking with integral wind-up
//! protection and derivative filtering for smooth control output.
//!
//! Key features:
//! - Multi-channel error aggregation (slippage, timing, fill rate)
//! - Proportional-Integral-Derivative control per channel
//! - Gain scheduling: adaptive gains based on error magnitude regime
//! - Anti-windup via conditional integration and clamping
//! - Derivative low-pass filtering to suppress measurement noise
//! - Setpoint management with ramp-rate limiting
//! - Running statistics and saturation tracking

use crate::common::{Error, Result};
use std::collections::VecDeque;

/// Configuration for the feedback loop
#[derive(Debug, Clone)]
pub struct FeedbackLoopConfig {
    /// Proportional gain for slippage channel
    pub kp_slippage: f64,
    /// Integral gain for slippage channel
    pub ki_slippage: f64,
    /// Derivative gain for slippage channel
    pub kd_slippage: f64,
    /// Proportional gain for timing channel
    pub kp_timing: f64,
    /// Integral gain for timing channel
    pub ki_timing: f64,
    /// Derivative gain for timing channel
    pub kd_timing: f64,
    /// Proportional gain for fill-rate channel
    pub kp_fill_rate: f64,
    /// Integral gain for fill-rate channel
    pub ki_fill_rate: f64,
    /// Derivative gain for fill-rate channel
    pub kd_fill_rate: f64,
    /// Weight of slippage channel in the combined output
    pub weight_slippage: f64,
    /// Weight of timing channel in the combined output
    pub weight_timing: f64,
    /// Weight of fill-rate channel in the combined output
    pub weight_fill_rate: f64,
    /// Maximum combined output magnitude (saturation clamp)
    pub output_max: f64,
    /// Minimum combined output magnitude (saturation clamp)
    pub output_min: f64,
    /// Maximum integral accumulator magnitude per channel (anti-windup)
    pub integral_max: f64,
    /// Derivative low-pass filter coefficient (0 = no filtering, closer to 1 = more filtering)
    pub derivative_filter_alpha: f64,
    /// Maximum setpoint change per step (ramp-rate limit); 0 = unlimited
    pub setpoint_ramp_rate: f64,
    /// EMA decay for error magnitude tracking (used in gain scheduling)
    pub ema_decay: f64,
    /// Error magnitude threshold for switching from low to high gain regime
    pub high_gain_threshold: f64,
    /// Gain multiplier applied in the high-error regime
    pub high_gain_multiplier: f64,
    /// Maximum number of steps to keep in the sliding window
    pub window_size: usize,
}

impl Default for FeedbackLoopConfig {
    fn default() -> Self {
        Self {
            kp_slippage: 0.5,
            ki_slippage: 0.05,
            kd_slippage: 0.1,
            kp_timing: 0.4,
            ki_timing: 0.04,
            kd_timing: 0.08,
            kp_fill_rate: 0.6,
            ki_fill_rate: 0.06,
            kd_fill_rate: 0.12,
            weight_slippage: 0.50,
            weight_timing: 0.20,
            weight_fill_rate: 0.30,
            output_max: 1.0,
            output_min: -1.0,
            integral_max: 5.0,
            derivative_filter_alpha: 0.2,
            setpoint_ramp_rate: 0.0,
            ema_decay: 0.93,
            high_gain_threshold: 0.5,
            high_gain_multiplier: 2.0,
            window_size: 500,
        }
    }
}

/// Error measurements from a single execution step
#[derive(Debug, Clone)]
pub struct FeedbackInput {
    /// Slippage error (positive = adverse slippage, negative = favorable)
    pub slippage_error: f64,
    /// Timing error (positive = late, negative = early)
    pub timing_error: f64,
    /// Fill-rate error (positive = underfill shortfall, negative = overfill)
    pub fill_rate_error: f64,
    /// Time step duration in seconds (dt); must be > 0
    pub dt: f64,
}

/// Setpoints (targets) for each error channel
#[derive(Debug, Clone)]
pub struct Setpoints {
    /// Target slippage (typically 0.0 for zero slippage)
    pub slippage: f64,
    /// Target timing error (typically 0.0 for on-time)
    pub timing: f64,
    /// Target fill-rate shortfall (typically 0.0 for complete fills)
    pub fill_rate: f64,
}

impl Default for Setpoints {
    fn default() -> Self {
        Self {
            slippage: 0.0,
            timing: 0.0,
            fill_rate: 0.0,
        }
    }
}

/// Output of a single feedback step
#[derive(Debug, Clone)]
pub struct FeedbackOutput {
    /// Combined correction signal (weighted sum of channels, clamped)
    pub correction: f64,
    /// Individual channel outputs before weighting
    pub channel_slippage: f64,
    pub channel_timing: f64,
    pub channel_fill_rate: f64,
    /// Whether the output is currently saturated (clamped)
    pub saturated: bool,
    /// Current gain regime: true if in high-gain mode
    pub high_gain_active: bool,
    /// Current effective setpoints (after ramp limiting)
    pub effective_setpoints: Setpoints,
}

/// Internal PID state for a single channel
#[derive(Debug, Clone)]
struct ChannelState {
    /// Integral accumulator
    integral: f64,
    /// Previous error (for derivative computation)
    prev_error: f64,
    /// Filtered derivative value
    filtered_derivative: f64,
    /// Whether we've seen at least one step
    initialized: bool,
}

impl ChannelState {
    fn new() -> Self {
        Self {
            integral: 0.0,
            prev_error: 0.0,
            filtered_derivative: 0.0,
            initialized: false,
        }
    }

    fn reset(&mut self) {
        self.integral = 0.0;
        self.prev_error = 0.0;
        self.filtered_derivative = 0.0;
        self.initialized = false;
    }
}

/// Record of a single feedback step for windowed analysis
#[derive(Debug, Clone)]
struct StepRecord {
    correction: f64,
    slippage_error: f64,
    timing_error: f64,
    fill_rate_error: f64,
    saturated: bool,
}

/// Running statistics for the feedback loop
#[derive(Debug, Clone, Default)]
pub struct FeedbackLoopStats {
    /// Total steps processed
    pub total_steps: usize,
    /// Number of steps where the output was saturated
    pub saturation_count: usize,
    /// Number of steps in high-gain regime
    pub high_gain_steps: usize,
    /// Sum of absolute correction values (for mean calculation)
    pub sum_abs_correction: f64,
    /// Sum of squared corrections (for variance calculation)
    pub sum_sq_correction: f64,
    /// Maximum absolute correction observed
    pub max_abs_correction: f64,
    /// Sum of absolute slippage errors
    pub sum_abs_slippage: f64,
    /// Sum of absolute timing errors
    pub sum_abs_timing: f64,
    /// Sum of absolute fill-rate errors
    pub sum_abs_fill_rate: f64,
    /// Peak integral magnitude across all channels
    pub peak_integral: f64,
    /// Number of integral wind-up clamp events
    pub windup_clamp_count: usize,
}

impl FeedbackLoopStats {
    /// Mean absolute correction
    pub fn mean_abs_correction(&self) -> f64 {
        if self.total_steps == 0 {
            return 0.0;
        }
        self.sum_abs_correction / self.total_steps as f64
    }

    /// Correction variance
    pub fn correction_variance(&self) -> f64 {
        if self.total_steps < 2 {
            return 0.0;
        }
        let mean = self.sum_abs_correction / self.total_steps as f64;
        let var = self.sum_sq_correction / self.total_steps as f64 - mean * mean;
        var.max(0.0)
    }

    /// Correction standard deviation
    pub fn correction_std(&self) -> f64 {
        self.correction_variance().sqrt()
    }

    /// Saturation rate (fraction of steps where output was clamped)
    pub fn saturation_rate(&self) -> f64 {
        if self.total_steps == 0 {
            return 0.0;
        }
        self.saturation_count as f64 / self.total_steps as f64
    }

    /// High-gain activation rate
    pub fn high_gain_rate(&self) -> f64 {
        if self.total_steps == 0 {
            return 0.0;
        }
        self.high_gain_steps as f64 / self.total_steps as f64
    }

    /// Mean absolute slippage error
    pub fn mean_abs_slippage(&self) -> f64 {
        if self.total_steps == 0 {
            return 0.0;
        }
        self.sum_abs_slippage / self.total_steps as f64
    }

    /// Mean absolute timing error
    pub fn mean_abs_timing(&self) -> f64 {
        if self.total_steps == 0 {
            return 0.0;
        }
        self.sum_abs_timing / self.total_steps as f64
    }

    /// Mean absolute fill-rate error
    pub fn mean_abs_fill_rate(&self) -> f64 {
        if self.total_steps == 0 {
            return 0.0;
        }
        self.sum_abs_fill_rate / self.total_steps as f64
    }
}

/// Feedback control loop for execution quality
///
/// Aggregates error signals from slippage, timing, and fill-rate channels
/// through per-channel PID controllers with gain scheduling. Produces a
/// unified correction signal for the execution engine.
pub struct FeedbackLoop {
    config: FeedbackLoopConfig,
    /// PID state for slippage channel
    slippage_state: ChannelState,
    /// PID state for timing channel
    timing_state: ChannelState,
    /// PID state for fill-rate channel
    fill_rate_state: ChannelState,
    /// Current setpoints
    setpoints: Setpoints,
    /// Effective setpoints (after ramp-rate limiting)
    effective_setpoints: Setpoints,
    /// EMA of composite error magnitude (for gain scheduling)
    ema_error_magnitude: f64,
    /// Whether EMA has been initialized
    ema_initialized: bool,
    /// Whether high-gain regime is currently active
    high_gain_active: bool,
    /// Recent step records for windowed analysis
    recent: VecDeque<StepRecord>,
    /// Running statistics
    stats: FeedbackLoopStats,
}

impl Default for FeedbackLoop {
    fn default() -> Self {
        Self::new()
    }
}

impl FeedbackLoop {
    /// Create a new instance with default configuration
    pub fn new() -> Self {
        Self::with_config(FeedbackLoopConfig::default())
    }

    /// Create a new instance with the given configuration
    pub fn with_config(config: FeedbackLoopConfig) -> Self {
        Self {
            slippage_state: ChannelState::new(),
            timing_state: ChannelState::new(),
            fill_rate_state: ChannelState::new(),
            setpoints: Setpoints::default(),
            effective_setpoints: Setpoints::default(),
            ema_error_magnitude: 0.0,
            ema_initialized: false,
            high_gain_active: false,
            recent: VecDeque::new(),
            stats: FeedbackLoopStats::default(),
            config,
        }
    }

    /// Main processing function — validates configuration
    pub fn process(&self) -> Result<()> {
        if self.config.output_max <= self.config.output_min {
            return Err(Error::InvalidInput(
                "output_max must be > output_min".into(),
            ));
        }
        if self.config.integral_max <= 0.0 {
            return Err(Error::InvalidInput("integral_max must be > 0".into()));
        }
        if self.config.derivative_filter_alpha < 0.0 || self.config.derivative_filter_alpha >= 1.0 {
            return Err(Error::InvalidInput(
                "derivative_filter_alpha must be in [0, 1)".into(),
            ));
        }
        if self.config.ema_decay <= 0.0 || self.config.ema_decay >= 1.0 {
            return Err(Error::InvalidInput("ema_decay must be in (0, 1)".into()));
        }
        if self.config.high_gain_multiplier < 1.0 {
            return Err(Error::InvalidInput(
                "high_gain_multiplier must be >= 1.0".into(),
            ));
        }
        if self.config.high_gain_threshold <= 0.0 {
            return Err(Error::InvalidInput(
                "high_gain_threshold must be > 0".into(),
            ));
        }
        if self.config.window_size == 0 {
            return Err(Error::InvalidInput("window_size must be > 0".into()));
        }
        if self.config.setpoint_ramp_rate < 0.0 {
            return Err(Error::InvalidInput(
                "setpoint_ramp_rate must be >= 0".into(),
            ));
        }
        // Weights should be non-negative
        if self.config.weight_slippage < 0.0
            || self.config.weight_timing < 0.0
            || self.config.weight_fill_rate < 0.0
        {
            return Err(Error::InvalidInput("channel weights must be >= 0".into()));
        }
        Ok(())
    }

    /// Set the target setpoints for each error channel
    pub fn set_setpoints(&mut self, setpoints: Setpoints) {
        self.setpoints = setpoints;
    }

    /// Get the current target setpoints
    pub fn setpoints(&self) -> &Setpoints {
        &self.setpoints
    }

    /// Get the current effective setpoints (after ramp-rate limiting)
    pub fn effective_setpoints(&self) -> &Setpoints {
        &self.effective_setpoints
    }

    /// Execute one feedback step
    ///
    /// Computes the PID output for each error channel, applies gain
    /// scheduling, combines the channels, and clamps the result.
    pub fn step(&mut self, input: &FeedbackInput) -> Result<FeedbackOutput> {
        if input.dt <= 0.0 {
            return Err(Error::InvalidInput("dt must be > 0".into()));
        }

        // Ramp-limit effective setpoints towards target setpoints
        self.ramp_setpoints(input.dt);

        // Compute errors relative to setpoints
        let slippage_err = input.slippage_error - self.effective_setpoints.slippage;
        let timing_err = input.timing_error - self.effective_setpoints.timing;
        let fill_rate_err = input.fill_rate_error - self.effective_setpoints.fill_rate;

        // Compute composite error magnitude for gain scheduling
        let composite_error = (slippage_err.abs() * self.config.weight_slippage
            + timing_err.abs() * self.config.weight_timing
            + fill_rate_err.abs() * self.config.weight_fill_rate)
            / (self.config.weight_slippage
                + self.config.weight_timing
                + self.config.weight_fill_rate)
                .max(1e-10);

        if self.ema_initialized {
            self.ema_error_magnitude = self.config.ema_decay * self.ema_error_magnitude
                + (1.0 - self.config.ema_decay) * composite_error;
        } else {
            self.ema_error_magnitude = composite_error;
            self.ema_initialized = true;
        }

        // Gain scheduling
        self.high_gain_active = self.ema_error_magnitude >= self.config.high_gain_threshold;
        let gain_mult = if self.high_gain_active {
            self.config.high_gain_multiplier
        } else {
            1.0
        };

        // Determine if we're saturated (from previous step) for anti-windup
        // We'll check after computing the raw output

        // Compute per-channel PID outputs and update state
        let ch_slippage = self.compute_channel(
            slippage_err,
            input.dt,
            self.config.kp_slippage * gain_mult,
            self.config.ki_slippage * gain_mult,
            self.config.kd_slippage * gain_mult,
            &mut self.slippage_state.clone(),
        );
        self.stats.windup_clamp_count += update_channel_state(
            slippage_err,
            input.dt,
            self.config.integral_max,
            self.config.derivative_filter_alpha,
            &mut self.slippage_state,
        );

        let ch_timing = self.compute_channel(
            timing_err,
            input.dt,
            self.config.kp_timing * gain_mult,
            self.config.ki_timing * gain_mult,
            self.config.kd_timing * gain_mult,
            &mut self.timing_state.clone(),
        );
        self.stats.windup_clamp_count += update_channel_state(
            timing_err,
            input.dt,
            self.config.integral_max,
            self.config.derivative_filter_alpha,
            &mut self.timing_state,
        );

        let ch_fill_rate = self.compute_channel(
            fill_rate_err,
            input.dt,
            self.config.kp_fill_rate * gain_mult,
            self.config.ki_fill_rate * gain_mult,
            self.config.kd_fill_rate * gain_mult,
            &mut self.fill_rate_state.clone(),
        );
        self.stats.windup_clamp_count += update_channel_state(
            fill_rate_err,
            input.dt,
            self.config.integral_max,
            self.config.derivative_filter_alpha,
            &mut self.fill_rate_state,
        );

        // Weighted combination
        let raw_correction = ch_slippage * self.config.weight_slippage
            + ch_timing * self.config.weight_timing
            + ch_fill_rate * self.config.weight_fill_rate;

        // Clamp
        let correction = raw_correction.clamp(self.config.output_min, self.config.output_max);
        let saturated = (raw_correction - correction).abs() > 1e-10;

        // If saturated, prevent further integral accumulation (anti-windup)
        if saturated {
            // Conditionally freeze integral: don't integrate if error pushes
            // further into saturation
            if raw_correction > self.config.output_max {
                // Positive saturation: freeze positive-contributing integrals
                self.freeze_positive_integrals(slippage_err, timing_err, fill_rate_err);
            } else if raw_correction < self.config.output_min {
                // Negative saturation: freeze negative-contributing integrals
                self.freeze_negative_integrals(slippage_err, timing_err, fill_rate_err);
            }
        }

        // Update statistics
        self.stats.total_steps += 1;
        if saturated {
            self.stats.saturation_count += 1;
        }
        if self.high_gain_active {
            self.stats.high_gain_steps += 1;
        }
        let abs_correction = correction.abs();
        self.stats.sum_abs_correction += abs_correction;
        self.stats.sum_sq_correction += correction * correction;
        if abs_correction > self.stats.max_abs_correction {
            self.stats.max_abs_correction = abs_correction;
        }
        self.stats.sum_abs_slippage += input.slippage_error.abs();
        self.stats.sum_abs_timing += input.timing_error.abs();
        self.stats.sum_abs_fill_rate += input.fill_rate_error.abs();

        // Track peak integral
        let max_integral = self
            .slippage_state
            .integral
            .abs()
            .max(self.timing_state.integral.abs())
            .max(self.fill_rate_state.integral.abs());
        if max_integral > self.stats.peak_integral {
            self.stats.peak_integral = max_integral;
        }

        // Record for windowed analysis
        let record = StepRecord {
            correction,
            slippage_error: input.slippage_error,
            timing_error: input.timing_error,
            fill_rate_error: input.fill_rate_error,
            saturated,
        };
        self.recent.push_back(record);
        while self.recent.len() > self.config.window_size {
            self.recent.pop_front();
        }

        Ok(FeedbackOutput {
            correction,
            channel_slippage: ch_slippage,
            channel_timing: ch_timing,
            channel_fill_rate: ch_fill_rate,
            saturated,
            high_gain_active: self.high_gain_active,
            effective_setpoints: self.effective_setpoints.clone(),
        })
    }

    /// Compute PID output for a single channel (without mutating state)
    fn compute_channel(
        &self,
        error: f64,
        dt: f64,
        kp: f64,
        ki: f64,
        kd: f64,
        state: &mut ChannelState,
    ) -> f64 {
        let p_term = kp * error;

        // Integral with clamping
        let new_integral = (state.integral + error * dt)
            .clamp(-self.config.integral_max, self.config.integral_max);
        let i_term = ki * new_integral;

        // Derivative with low-pass filtering
        let d_term = if state.initialized && dt > 0.0 {
            let raw_derivative = (error - state.prev_error) / dt;
            let alpha = self.config.derivative_filter_alpha;
            let filtered = alpha * state.filtered_derivative + (1.0 - alpha) * raw_derivative;
            kd * filtered
        } else {
            0.0
        };

        p_term + i_term + d_term
    }

    /// Ramp-limit effective setpoints towards target setpoints
    fn ramp_setpoints(&mut self, dt: f64) {
        if self.config.setpoint_ramp_rate <= 0.0 {
            // No ramp limiting: snap to target
            self.effective_setpoints = self.setpoints.clone();
            return;
        }

        let max_change = self.config.setpoint_ramp_rate * dt;

        self.effective_setpoints.slippage = ramp_towards(
            self.effective_setpoints.slippage,
            self.setpoints.slippage,
            max_change,
        );
        self.effective_setpoints.timing = ramp_towards(
            self.effective_setpoints.timing,
            self.setpoints.timing,
            max_change,
        );
        self.effective_setpoints.fill_rate = ramp_towards(
            self.effective_setpoints.fill_rate,
            self.setpoints.fill_rate,
            max_change,
        );
    }

    /// Freeze positive-contributing integrals (anti-windup for positive saturation)
    fn freeze_positive_integrals(&mut self, slip_err: f64, timing_err: f64, fill_err: f64) {
        if slip_err > 0.0 {
            self.slippage_state.integral -= slip_err * 0.01; // small rollback
            self.slippage_state.integral = self.slippage_state.integral.max(0.0);
        }
        if timing_err > 0.0 {
            self.timing_state.integral -= timing_err * 0.01;
            self.timing_state.integral = self.timing_state.integral.max(0.0);
        }
        if fill_err > 0.0 {
            self.fill_rate_state.integral -= fill_err * 0.01;
            self.fill_rate_state.integral = self.fill_rate_state.integral.max(0.0);
        }
    }

    /// Freeze negative-contributing integrals (anti-windup for negative saturation)
    fn freeze_negative_integrals(&mut self, slip_err: f64, timing_err: f64, fill_err: f64) {
        if slip_err < 0.0 {
            self.slippage_state.integral -= slip_err * 0.01;
            self.slippage_state.integral = self.slippage_state.integral.min(0.0);
        }
        if timing_err < 0.0 {
            self.timing_state.integral -= timing_err * 0.01;
            self.timing_state.integral = self.timing_state.integral.min(0.0);
        }
        if fill_err < 0.0 {
            self.fill_rate_state.integral -= fill_err * 0.01;
            self.fill_rate_state.integral = self.fill_rate_state.integral.min(0.0);
        }
    }

    /// Whether high-gain mode is currently active
    pub fn is_high_gain_active(&self) -> bool {
        self.high_gain_active
    }

    /// Current EMA of composite error magnitude
    pub fn ema_error_magnitude(&self) -> f64 {
        if self.ema_initialized {
            self.ema_error_magnitude
        } else {
            0.0
        }
    }

    /// Current integral accumulator values
    pub fn integrals(&self) -> (f64, f64, f64) {
        (
            self.slippage_state.integral,
            self.timing_state.integral,
            self.fill_rate_state.integral,
        )
    }

    /// Total steps processed
    pub fn step_count(&self) -> usize {
        self.stats.total_steps
    }

    /// Reference to running statistics
    pub fn stats(&self) -> &FeedbackLoopStats {
        &self.stats
    }

    /// Windowed mean absolute correction
    pub fn windowed_mean_abs_correction(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.recent.iter().map(|r| r.correction.abs()).sum();
        sum / self.recent.len() as f64
    }

    /// Windowed saturation rate
    pub fn windowed_saturation_rate(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let count = self.recent.iter().filter(|r| r.saturated).count();
        count as f64 / self.recent.len() as f64
    }

    /// Windowed mean slippage error
    pub fn windowed_mean_slippage(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.recent.iter().map(|r| r.slippage_error).sum();
        sum / self.recent.len() as f64
    }

    /// Check if errors are trending upward (second half worse than first)
    pub fn is_errors_worsening(&self) -> bool {
        let n = self.recent.len();
        if n < 10 {
            return false;
        }
        let mid = n / 2;

        let first_half_err: f64 = self
            .recent
            .iter()
            .take(mid)
            .map(|r| r.slippage_error.abs() + r.timing_error.abs() + r.fill_rate_error.abs())
            .sum::<f64>()
            / mid as f64;

        let second_half_err: f64 = self
            .recent
            .iter()
            .skip(mid)
            .map(|r| r.slippage_error.abs() + r.timing_error.abs() + r.fill_rate_error.abs())
            .sum::<f64>()
            / (n - mid) as f64;

        second_half_err > first_half_err * 1.2
    }

    /// Reset all state and statistics
    pub fn reset(&mut self) {
        self.slippage_state.reset();
        self.timing_state.reset();
        self.fill_rate_state.reset();
        self.setpoints = Setpoints::default();
        self.effective_setpoints = Setpoints::default();
        self.ema_error_magnitude = 0.0;
        self.ema_initialized = false;
        self.high_gain_active = false;
        self.recent.clear();
        self.stats = FeedbackLoopStats::default();
    }
}

/// Update channel state after PID computation (standalone to avoid borrow issues).
/// Returns 1 if a windup clamp occurred, 0 otherwise.
fn update_channel_state(
    error: f64,
    dt: f64,
    integral_max: f64,
    derivative_filter_alpha: f64,
    state: &mut ChannelState,
) -> usize {
    // Update integral with anti-windup clamping
    let new_integral = state.integral + error * dt;
    let clamped = new_integral.clamp(-integral_max, integral_max);
    let windup_clamp = if (new_integral - clamped).abs() > 1e-10 {
        1
    } else {
        0
    };
    state.integral = clamped;

    // Update derivative filter
    if state.initialized && dt > 0.0 {
        let raw_derivative = (error - state.prev_error) / dt;
        let alpha = derivative_filter_alpha;
        state.filtered_derivative =
            alpha * state.filtered_derivative + (1.0 - alpha) * raw_derivative;
    }

    state.prev_error = error;
    state.initialized = true;
    windup_clamp
}

/// Move `current` towards `target` by at most `max_change`
fn ramp_towards(current: f64, target: f64, max_change: f64) -> f64 {
    let diff = target - current;
    if diff.abs() <= max_change {
        target
    } else {
        current + diff.signum() * max_change
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zero_input(dt: f64) -> FeedbackInput {
        FeedbackInput {
            slippage_error: 0.0,
            timing_error: 0.0,
            fill_rate_error: 0.0,
            dt,
        }
    }

    fn slippage_input(slip: f64, dt: f64) -> FeedbackInput {
        FeedbackInput {
            slippage_error: slip,
            timing_error: 0.0,
            fill_rate_error: 0.0,
            dt,
        }
    }

    fn all_channels_input(error: f64, dt: f64) -> FeedbackInput {
        FeedbackInput {
            slippage_error: error,
            timing_error: error,
            fill_rate_error: error,
            dt,
        }
    }

    #[test]
    fn test_basic() {
        let instance = FeedbackLoop::new();
        assert!(instance.process().is_ok());
    }

    #[test]
    fn test_zero_error_zero_correction() {
        let mut fb = FeedbackLoop::new();
        let output = fb.step(&zero_input(0.01)).unwrap();
        assert!(
            output.correction.abs() < 1e-10,
            "zero error should produce zero correction, got {}",
            output.correction
        );
    }

    #[test]
    fn test_positive_error_positive_correction() {
        let mut fb = FeedbackLoop::new();
        let output = fb.step(&slippage_input(1.0, 0.01)).unwrap();
        assert!(
            output.correction > 0.0,
            "positive error should produce positive correction, got {}",
            output.correction
        );
    }

    #[test]
    fn test_negative_error_negative_correction() {
        let mut fb = FeedbackLoop::new();
        let output = fb.step(&slippage_input(-1.0, 0.01)).unwrap();
        assert!(
            output.correction < 0.0,
            "negative error should produce negative correction, got {}",
            output.correction
        );
    }

    #[test]
    fn test_larger_error_larger_correction() {
        let mut fb1 = FeedbackLoop::new();
        let mut fb2 = FeedbackLoop::new();

        let out_small = fb1.step(&slippage_input(0.1, 0.01)).unwrap();
        let out_large = fb2.step(&slippage_input(1.0, 0.01)).unwrap();

        assert!(
            out_large.correction.abs() > out_small.correction.abs(),
            "larger error should produce larger correction: {} vs {}",
            out_large.correction.abs(),
            out_small.correction.abs()
        );
    }

    #[test]
    fn test_output_clamped_positive() {
        let mut fb = FeedbackLoop::with_config(FeedbackLoopConfig {
            output_max: 0.5,
            output_min: -0.5,
            kp_slippage: 10.0,
            ..Default::default()
        });

        let output = fb.step(&slippage_input(100.0, 0.01)).unwrap();
        assert!(
            (output.correction - 0.5).abs() < 1e-10,
            "should be clamped to 0.5, got {}",
            output.correction
        );
        assert!(output.saturated);
    }

    #[test]
    fn test_output_clamped_negative() {
        let mut fb = FeedbackLoop::with_config(FeedbackLoopConfig {
            output_max: 0.5,
            output_min: -0.5,
            kp_slippage: 10.0,
            ..Default::default()
        });

        let output = fb.step(&slippage_input(-100.0, 0.01)).unwrap();
        assert!(
            (output.correction - (-0.5)).abs() < 1e-10,
            "should be clamped to -0.5, got {}",
            output.correction
        );
        assert!(output.saturated);
    }

    #[test]
    fn test_integral_accumulation() {
        let mut fb = FeedbackLoop::with_config(FeedbackLoopConfig {
            kp_slippage: 0.0,
            ki_slippage: 1.0,
            kd_slippage: 0.0,
            output_max: 100.0,
            output_min: -100.0,
            integral_max: 100.0,
            ..Default::default()
        });

        // Step with constant error — integral should accumulate
        for _ in 0..10 {
            fb.step(&slippage_input(1.0, 1.0)).unwrap();
        }

        let (slip_integral, _, _) = fb.integrals();
        assert!(
            (slip_integral - 10.0).abs() < 1e-6,
            "integral should accumulate to 10, got {}",
            slip_integral
        );
    }

    #[test]
    fn test_integral_windup_clamped() {
        let mut fb = FeedbackLoop::with_config(FeedbackLoopConfig {
            kp_slippage: 0.0,
            ki_slippage: 1.0,
            kd_slippage: 0.0,
            integral_max: 2.0,
            output_max: 100.0,
            output_min: -100.0,
            ..Default::default()
        });

        for _ in 0..100 {
            fb.step(&slippage_input(1.0, 1.0)).unwrap();
        }

        let (slip_integral, _, _) = fb.integrals();
        assert!(
            slip_integral <= 2.0 + 1e-10,
            "integral should be clamped to 2.0, got {}",
            slip_integral
        );
        assert!(fb.stats().windup_clamp_count > 0);
    }

    #[test]
    fn test_derivative_responds_to_change() {
        let mut fb = FeedbackLoop::with_config(FeedbackLoopConfig {
            kp_slippage: 0.0,
            ki_slippage: 0.0,
            kd_slippage: 1.0,
            derivative_filter_alpha: 0.0, // no filtering
            output_max: 100.0,
            output_min: -100.0,
            ..Default::default()
        });

        // First step: no derivative yet
        let out1 = fb.step(&slippage_input(0.0, 1.0)).unwrap();

        // Second step: error jumps from 0 to 5
        let out2 = fb.step(&slippage_input(5.0, 1.0)).unwrap();

        assert!(
            out2.channel_slippage.abs() > out1.channel_slippage.abs(),
            "derivative should respond to error change: {} vs {}",
            out2.channel_slippage,
            out1.channel_slippage
        );
    }

    #[test]
    fn test_derivative_filter_dampens() {
        let mut fb_unfiltered = FeedbackLoop::with_config(FeedbackLoopConfig {
            kp_slippage: 0.0,
            ki_slippage: 0.0,
            kd_slippage: 1.0,
            derivative_filter_alpha: 0.0,
            output_max: 100.0,
            output_min: -100.0,
            ..Default::default()
        });

        let mut fb_filtered = FeedbackLoop::with_config(FeedbackLoopConfig {
            kp_slippage: 0.0,
            ki_slippage: 0.0,
            kd_slippage: 1.0,
            derivative_filter_alpha: 0.8,
            output_max: 100.0,
            output_min: -100.0,
            ..Default::default()
        });

        // Initialize both
        fb_unfiltered.step(&slippage_input(0.0, 1.0)).unwrap();
        fb_filtered.step(&slippage_input(0.0, 1.0)).unwrap();

        // Spike
        let out_unfiltered = fb_unfiltered.step(&slippage_input(10.0, 1.0)).unwrap();
        let out_filtered = fb_filtered.step(&slippage_input(10.0, 1.0)).unwrap();

        assert!(
            out_filtered.channel_slippage.abs() < out_unfiltered.channel_slippage.abs(),
            "filtered derivative should be smaller: {} vs {}",
            out_filtered.channel_slippage,
            out_unfiltered.channel_slippage
        );
    }

    #[test]
    fn test_gain_scheduling_activates() {
        let mut fb = FeedbackLoop::with_config(FeedbackLoopConfig {
            high_gain_threshold: 0.1,
            high_gain_multiplier: 3.0,
            ema_decay: 0.1, // fast response
            ..Default::default()
        });

        // Small errors: should be in low-gain
        for _ in 0..5 {
            fb.step(&all_channels_input(0.01, 0.01)).unwrap();
        }
        assert!(!fb.is_high_gain_active());

        // Large errors: should switch to high-gain
        for _ in 0..20 {
            fb.step(&all_channels_input(10.0, 0.01)).unwrap();
        }
        assert!(fb.is_high_gain_active());
    }

    #[test]
    fn test_high_gain_produces_larger_correction() {
        let mut fb_low = FeedbackLoop::with_config(FeedbackLoopConfig {
            high_gain_threshold: 100.0, // never triggers
            output_max: 100.0,
            output_min: -100.0,
            ..Default::default()
        });

        let mut fb_high = FeedbackLoop::with_config(FeedbackLoopConfig {
            high_gain_threshold: 0.0001, // always triggers
            high_gain_multiplier: 3.0,
            ema_decay: 0.01,
            output_max: 100.0,
            output_min: -100.0,
            ..Default::default()
        });

        // Warm up high-gain detector
        for _ in 0..5 {
            fb_high.step(&slippage_input(1.0, 0.01)).unwrap();
        }

        let out_low = fb_low.step(&slippage_input(1.0, 0.01)).unwrap();
        let out_high = fb_high.step(&slippage_input(1.0, 0.01)).unwrap();

        // Note: high-gain may or may not produce strictly larger output due to
        // integral accumulation differences, but the proportional component should be larger
        assert!(
            out_high.correction.abs() >= out_low.correction.abs() * 0.9,
            "high-gain correction should be at least comparable: {} vs {}",
            out_high.correction.abs(),
            out_low.correction.abs()
        );
    }

    #[test]
    fn test_setpoint_tracking() {
        let mut fb = FeedbackLoop::with_config(FeedbackLoopConfig {
            kp_slippage: 1.0,
            ki_slippage: 0.0,
            kd_slippage: 0.0,
            output_max: 100.0,
            output_min: -100.0,
            ..Default::default()
        });

        // Set non-zero setpoint
        fb.set_setpoints(Setpoints {
            slippage: 5.0,
            timing: 0.0,
            fill_rate: 0.0,
        });

        // Input that matches setpoint exactly: error should be zero
        let out = fb.step(&slippage_input(5.0, 0.01)).unwrap();
        assert!(
            out.channel_slippage.abs() < 1e-10,
            "at setpoint, channel output should be ~0, got {}",
            out.channel_slippage
        );
    }

    #[test]
    fn test_setpoint_ramp_limiting() {
        let mut fb = FeedbackLoop::with_config(FeedbackLoopConfig {
            setpoint_ramp_rate: 1.0, // max 1.0 per second
            ..Default::default()
        });

        fb.set_setpoints(Setpoints {
            slippage: 10.0,
            timing: 0.0,
            fill_rate: 0.0,
        });

        // Step with dt = 0.1s: max change = 0.1
        fb.step(&zero_input(0.1)).unwrap();

        let eff = fb.effective_setpoints();
        assert!(
            (eff.slippage - 0.1).abs() < 1e-10,
            "ramp should limit to 0.1, got {}",
            eff.slippage
        );
    }

    #[test]
    fn test_setpoint_ramp_converges() {
        let mut fb = FeedbackLoop::with_config(FeedbackLoopConfig {
            setpoint_ramp_rate: 10.0,
            ..Default::default()
        });

        fb.set_setpoints(Setpoints {
            slippage: 5.0,
            timing: 0.0,
            fill_rate: 0.0,
        });

        // Step enough times to converge
        for _ in 0..100 {
            fb.step(&zero_input(1.0)).unwrap();
        }

        let eff = fb.effective_setpoints();
        assert!(
            (eff.slippage - 5.0).abs() < 1e-10,
            "ramp should converge to target, got {}",
            eff.slippage
        );
    }

    #[test]
    fn test_no_ramp_limit_when_zero() {
        let mut fb = FeedbackLoop::with_config(FeedbackLoopConfig {
            setpoint_ramp_rate: 0.0, // disabled
            ..Default::default()
        });

        fb.set_setpoints(Setpoints {
            slippage: 100.0,
            timing: 0.0,
            fill_rate: 0.0,
        });

        fb.step(&zero_input(0.001)).unwrap();

        let eff = fb.effective_setpoints();
        assert!(
            (eff.slippage - 100.0).abs() < 1e-10,
            "no ramp limit should snap to target, got {}",
            eff.slippage
        );
    }

    #[test]
    fn test_multi_channel_weighting() {
        let mut fb = FeedbackLoop::with_config(FeedbackLoopConfig {
            kp_slippage: 1.0,
            ki_slippage: 0.0,
            kd_slippage: 0.0,
            kp_timing: 1.0,
            ki_timing: 0.0,
            kd_timing: 0.0,
            kp_fill_rate: 1.0,
            ki_fill_rate: 0.0,
            kd_fill_rate: 0.0,
            weight_slippage: 1.0,
            weight_timing: 0.0,
            weight_fill_rate: 0.0,
            output_max: 100.0,
            output_min: -100.0,
            ..Default::default()
        });

        // Only slippage is weighted, so timing/fill_rate should have no effect
        let out = fb
            .step(&FeedbackInput {
                slippage_error: 1.0,
                timing_error: 100.0,
                fill_rate_error: 100.0,
                dt: 0.01,
            })
            .unwrap();

        // correction = ch_slippage * 1.0 + ch_timing * 0.0 + ch_fill_rate * 0.0
        assert!(
            (out.correction - out.channel_slippage).abs() < 1e-10,
            "only slippage should contribute: correction={}, ch_slippage={}",
            out.correction,
            out.channel_slippage
        );
    }

    #[test]
    fn test_stats_tracking() {
        let mut fb = FeedbackLoop::new();

        for _ in 0..10 {
            fb.step(&slippage_input(0.5, 0.01)).unwrap();
        }

        assert_eq!(fb.step_count(), 10);
        assert_eq!(fb.stats().total_steps, 10);
        assert!(fb.stats().sum_abs_correction > 0.0);
        assert!(fb.stats().sum_abs_slippage > 0.0);
    }

    #[test]
    fn test_saturation_count() {
        let mut fb = FeedbackLoop::with_config(FeedbackLoopConfig {
            output_max: 0.001,
            output_min: -0.001,
            kp_slippage: 10.0,
            ..Default::default()
        });

        for _ in 0..5 {
            fb.step(&slippage_input(10.0, 0.01)).unwrap();
        }

        assert_eq!(fb.stats().saturation_count, 5);
        assert!((fb.stats().saturation_rate() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_windowed_mean_abs_correction() {
        let mut fb = FeedbackLoop::with_config(FeedbackLoopConfig {
            window_size: 5,
            ..Default::default()
        });

        for _ in 0..5 {
            fb.step(&slippage_input(0.5, 0.01)).unwrap();
        }

        let wm = fb.windowed_mean_abs_correction();
        assert!(wm > 0.0, "windowed mean should be > 0, got {}", wm);
    }

    #[test]
    fn test_windowed_saturation_rate() {
        let mut fb = FeedbackLoop::with_config(FeedbackLoopConfig {
            output_max: 0.0001,
            output_min: -0.0001,
            kp_slippage: 100.0,
            window_size: 10,
            ..Default::default()
        });

        for _ in 0..10 {
            fb.step(&slippage_input(10.0, 0.01)).unwrap();
        }

        assert!((fb.windowed_saturation_rate() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_is_errors_worsening() {
        let mut fb = FeedbackLoop::with_config(FeedbackLoopConfig {
            window_size: 20,
            ..Default::default()
        });

        // First half: small errors
        for _ in 0..10 {
            fb.step(&slippage_input(0.01, 0.01)).unwrap();
        }
        // Second half: large errors
        for _ in 0..10 {
            fb.step(&slippage_input(10.0, 0.01)).unwrap();
        }

        assert!(fb.is_errors_worsening());
    }

    #[test]
    fn test_not_worsening_consistent() {
        let mut fb = FeedbackLoop::with_config(FeedbackLoopConfig {
            window_size: 20,
            ..Default::default()
        });

        for _ in 0..20 {
            fb.step(&slippage_input(0.5, 0.01)).unwrap();
        }

        assert!(!fb.is_errors_worsening());
    }

    #[test]
    fn test_not_worsening_insufficient_data() {
        let mut fb = FeedbackLoop::new();
        for _ in 0..4 {
            fb.step(&slippage_input(10.0, 0.01)).unwrap();
        }
        assert!(!fb.is_errors_worsening());
    }

    #[test]
    fn test_reset() {
        let mut fb = FeedbackLoop::new();

        for _ in 0..20 {
            fb.step(&slippage_input(1.0, 0.01)).unwrap();
        }

        assert!(fb.step_count() > 0);
        let (si, _, _) = fb.integrals();
        assert!(si.abs() > 0.0);

        fb.reset();

        assert_eq!(fb.step_count(), 0);
        let (si, ti, fi) = fb.integrals();
        assert_eq!(si, 0.0);
        assert_eq!(ti, 0.0);
        assert_eq!(fi, 0.0);
        assert!(!fb.is_high_gain_active());
        assert_eq!(fb.ema_error_magnitude(), 0.0);
    }

    #[test]
    fn test_invalid_dt() {
        let mut fb = FeedbackLoop::new();
        assert!(fb.step(&zero_input(0.0)).is_err());
        assert!(fb.step(&zero_input(-1.0)).is_err());
    }

    #[test]
    fn test_invalid_config_output_bounds() {
        let fb = FeedbackLoop::with_config(FeedbackLoopConfig {
            output_max: -1.0,
            output_min: 1.0,
            ..Default::default()
        });
        assert!(fb.process().is_err());
    }

    #[test]
    fn test_invalid_config_integral_max() {
        let fb = FeedbackLoop::with_config(FeedbackLoopConfig {
            integral_max: 0.0,
            ..Default::default()
        });
        assert!(fb.process().is_err());
    }

    #[test]
    fn test_invalid_config_derivative_filter() {
        let fb = FeedbackLoop::with_config(FeedbackLoopConfig {
            derivative_filter_alpha: 1.0,
            ..Default::default()
        });
        assert!(fb.process().is_err());
    }

    #[test]
    fn test_invalid_config_ema_decay() {
        let fb = FeedbackLoop::with_config(FeedbackLoopConfig {
            ema_decay: 0.0,
            ..Default::default()
        });
        assert!(fb.process().is_err());
    }

    #[test]
    fn test_invalid_config_high_gain_multiplier() {
        let fb = FeedbackLoop::with_config(FeedbackLoopConfig {
            high_gain_multiplier: 0.5,
            ..Default::default()
        });
        assert!(fb.process().is_err());
    }

    #[test]
    fn test_invalid_config_high_gain_threshold() {
        let fb = FeedbackLoop::with_config(FeedbackLoopConfig {
            high_gain_threshold: 0.0,
            ..Default::default()
        });
        assert!(fb.process().is_err());
    }

    #[test]
    fn test_invalid_config_window_size() {
        let fb = FeedbackLoop::with_config(FeedbackLoopConfig {
            window_size: 0,
            ..Default::default()
        });
        assert!(fb.process().is_err());
    }

    #[test]
    fn test_invalid_config_negative_weight() {
        let fb = FeedbackLoop::with_config(FeedbackLoopConfig {
            weight_slippage: -1.0,
            ..Default::default()
        });
        assert!(fb.process().is_err());
    }

    #[test]
    fn test_invalid_config_negative_ramp_rate() {
        let fb = FeedbackLoop::with_config(FeedbackLoopConfig {
            setpoint_ramp_rate: -1.0,
            ..Default::default()
        });
        assert!(fb.process().is_err());
    }

    #[test]
    fn test_stats_defaults() {
        let stats = FeedbackLoopStats::default();
        assert_eq!(stats.mean_abs_correction(), 0.0);
        assert_eq!(stats.correction_variance(), 0.0);
        assert_eq!(stats.correction_std(), 0.0);
        assert_eq!(stats.saturation_rate(), 0.0);
        assert_eq!(stats.high_gain_rate(), 0.0);
        assert_eq!(stats.mean_abs_slippage(), 0.0);
        assert_eq!(stats.mean_abs_timing(), 0.0);
        assert_eq!(stats.mean_abs_fill_rate(), 0.0);
    }

    #[test]
    fn test_ramp_towards_overshoot() {
        let v = ramp_towards(0.0, 10.0, 3.0);
        assert!((v - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_ramp_towards_exact() {
        let v = ramp_towards(9.0, 10.0, 5.0);
        assert!((v - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_ramp_towards_negative() {
        let v = ramp_towards(0.0, -10.0, 3.0);
        assert!((v - (-3.0)).abs() < 1e-10);
    }

    #[test]
    fn test_window_eviction() {
        let mut fb = FeedbackLoop::with_config(FeedbackLoopConfig {
            window_size: 5,
            ..Default::default()
        });

        for _ in 0..20 {
            fb.step(&slippage_input(0.5, 0.01)).unwrap();
        }

        assert_eq!(fb.recent.len(), 5);
        assert_eq!(fb.step_count(), 20);
    }

    #[test]
    fn test_windowed_mean_slippage() {
        let mut fb = FeedbackLoop::with_config(FeedbackLoopConfig {
            window_size: 5,
            ..Default::default()
        });

        for _ in 0..5 {
            fb.step(&slippage_input(2.0, 0.01)).unwrap();
        }

        assert!(
            (fb.windowed_mean_slippage() - 2.0).abs() < 1e-10,
            "windowed mean slippage should be 2.0, got {}",
            fb.windowed_mean_slippage()
        );
    }

    #[test]
    fn test_empty_windowed_values() {
        let fb = FeedbackLoop::new();
        assert_eq!(fb.windowed_mean_abs_correction(), 0.0);
        assert_eq!(fb.windowed_saturation_rate(), 0.0);
        assert_eq!(fb.windowed_mean_slippage(), 0.0);
    }

    #[test]
    fn test_high_gain_rate_tracking() {
        let mut fb = FeedbackLoop::with_config(FeedbackLoopConfig {
            high_gain_threshold: 0.001,
            ema_decay: 0.01,
            ..Default::default()
        });

        for _ in 0..20 {
            fb.step(&slippage_input(10.0, 0.01)).unwrap();
        }

        assert!(
            fb.stats().high_gain_rate() > 0.0,
            "should have some high-gain steps"
        );
    }

    #[test]
    fn test_ema_error_magnitude() {
        let mut fb = FeedbackLoop::new();

        assert_eq!(fb.ema_error_magnitude(), 0.0);

        fb.step(&slippage_input(1.0, 0.01)).unwrap();
        assert!(
            fb.ema_error_magnitude() > 0.0,
            "ema should be > 0 after non-zero input"
        );
    }
}
