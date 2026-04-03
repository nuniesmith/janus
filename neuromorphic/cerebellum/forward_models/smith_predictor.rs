//! Smith Predictor — dead-time compensator for execution control loops
//!
//! Part of the Cerebellum region
//! Component: forward_models
//!
//! Implements a Smith Predictor to compensate for the destabilising effect
//! of communication latency (dead time) in execution control loops. The
//! predictor maintains an internal model of the plant (market response)
//! and uses it to cancel the delay, allowing the controller to act as if
//! the loop were delay-free.
//!
//! Key features:
//! - Configurable dead-time model with adaptive latency estimation
//! - Internal plant model predicting market response to execution actions
//! - Delay buffer ring for time-shifted model predictions
//! - Mismatch detection between predicted and actual plant output
//! - EMA-smoothed model error tracking for online model adaptation
//! - Running statistics for prediction quality assessment

use crate::common::{Error, Result};
use std::collections::VecDeque;

/// Configuration for the Smith Predictor
#[derive(Debug, Clone)]
pub struct SmithPredictorConfig {
    /// Nominal dead time in milliseconds (communication round-trip latency)
    pub dead_time_ms: f64,
    /// Plant model gain: how much the plant output changes per unit of control input
    pub plant_gain: f64,
    /// Plant model time constant (first-order lag, in milliseconds)
    pub plant_time_constant_ms: f64,
    /// EMA decay factor for model error tracking (0 < decay < 1)
    pub ema_decay: f64,
    /// Maximum number of entries in the delay buffer
    pub delay_buffer_size: usize,
    /// Time step for discretisation (milliseconds)
    pub dt_ms: f64,
    /// Mismatch threshold: ratio of model error to signal magnitude that triggers alarm
    pub mismatch_threshold: f64,
    /// Minimum observations before mismatch detection activates
    pub min_samples: usize,
    /// Maximum number of observations in the sliding window for stats
    pub window_size: usize,
    /// Adaptive dead-time EMA decay (for learning actual latency)
    pub latency_ema_decay: f64,
}

impl Default for SmithPredictorConfig {
    fn default() -> Self {
        Self {
            dead_time_ms: 50.0,
            plant_gain: 1.0,
            plant_time_constant_ms: 100.0,
            ema_decay: 0.93,
            delay_buffer_size: 500,
            dt_ms: 1.0,
            mismatch_threshold: 0.3,
            min_samples: 20,
            window_size: 500,
            latency_ema_decay: 0.95,
        }
    }
}

/// A control action submitted to the plant (market)
#[derive(Debug, Clone)]
pub struct ControlAction {
    /// The control signal magnitude (e.g. order rate, execution speed)
    pub signal: f64,
    /// Timestamp in milliseconds when the action was issued
    pub timestamp_ms: f64,
}

/// An observed plant output (actual market response)
#[derive(Debug, Clone)]
pub struct PlantOutput {
    /// The observed plant output (e.g. actual fill rate, price movement)
    pub value: f64,
    /// Timestamp in milliseconds when the output was observed
    pub timestamp_ms: f64,
}

/// Result of the Smith Predictor compensation step
#[derive(Debug, Clone)]
pub struct CompensatedOutput {
    /// The compensated feedback signal (plant output minus delayed model prediction
    /// plus instantaneous model prediction) — used by the controller
    pub compensated_signal: f64,
    /// Instantaneous model prediction (no delay)
    pub model_prediction: f64,
    /// Delayed model prediction (shifted by dead time)
    pub delayed_prediction: f64,
    /// Raw plant output
    pub plant_output: f64,
    /// Current model error (plant output - delayed prediction)
    pub model_error: f64,
    /// Whether a model mismatch has been detected
    pub mismatch_detected: bool,
    /// Effective dead time currently used (ms)
    pub effective_dead_time_ms: f64,
}

/// Observation record for tracking prediction quality
#[derive(Debug, Clone)]
struct PredictionRecord {
    /// What the model predicted (delayed)
    predicted: f64,
    /// What was actually observed
    actual: f64,
    /// Squared error
    sq_error: f64,
}

/// Running statistics for the Smith Predictor
#[derive(Debug, Clone, Default)]
pub struct SmithPredictorStats {
    /// Total control actions processed
    pub total_actions: usize,
    /// Total plant outputs observed
    pub total_outputs: usize,
    /// Total compensation steps executed
    pub total_compensations: usize,
    /// Sum of squared model errors (for MSE calculation)
    pub sum_sq_error: f64,
    /// Sum of absolute model errors (for MAE calculation)
    pub sum_abs_error: f64,
    /// Sum of signed model errors (for bias calculation)
    pub sum_signed_error: f64,
    /// Maximum absolute model error observed
    pub max_abs_error: f64,
    /// Number of mismatch events detected
    pub mismatch_events: usize,
    /// Peak model error magnitude
    pub peak_error: f64,
    /// Current effective dead time (ms)
    pub effective_dead_time_ms: f64,
    /// Number of dead-time adaptation updates
    pub latency_updates: usize,
}

impl SmithPredictorStats {
    /// Mean squared error of model predictions
    pub fn mse(&self) -> f64 {
        if self.total_compensations == 0 {
            return 0.0;
        }
        self.sum_sq_error / self.total_compensations as f64
    }

    /// Root mean squared error
    pub fn rmse(&self) -> f64 {
        self.mse().sqrt()
    }

    /// Mean absolute error
    pub fn mae(&self) -> f64 {
        if self.total_compensations == 0 {
            return 0.0;
        }
        self.sum_abs_error / self.total_compensations as f64
    }

    /// Bias (mean signed error): positive = model over-predicts
    pub fn bias(&self) -> f64 {
        if self.total_compensations == 0 {
            return 0.0;
        }
        self.sum_signed_error / self.total_compensations as f64
    }
}

/// Internal first-order plant model state
#[derive(Debug, Clone)]
struct PlantModel {
    /// Current model output
    output: f64,
    /// Plant gain
    gain: f64,
    /// Time constant (ms)
    time_constant_ms: f64,
}

impl PlantModel {
    fn new(gain: f64, time_constant_ms: f64) -> Self {
        Self {
            output: 0.0,
            gain,
            time_constant_ms,
        }
    }

    /// Step the first-order model: y[k+1] = y[k] + (dt/tau) * (K*u - y[k])
    fn step(&mut self, input: f64, dt_ms: f64) -> f64 {
        if self.time_constant_ms <= 0.0 {
            // Instantaneous response
            self.output = self.gain * input;
        } else {
            let alpha = (dt_ms / self.time_constant_ms).min(1.0);
            self.output += alpha * (self.gain * input - self.output);
        }
        self.output
    }

    fn reset(&mut self) {
        self.output = 0.0;
    }
}

/// Smith Predictor — dead-time compensator for execution control loops
///
/// Maintains an internal model of the plant and a delay buffer to cancel
/// the effect of communication latency on the feedback loop. The compensated
/// signal allows the controller to operate as if there were no dead time.
///
/// Architecture:
/// ```text
///                    ┌──────────────┐
///   u ──────┬───────►│   Plant      ├───► y_actual
///           │        │  (+ delay)   │       │
///           │        └──────────────┘       │
///           │                               │
///           │   ┌──────────────┐            │
///           ├──►│ Plant Model  ├──► ŷ_inst  │
///           │   │  (no delay)  │            │
///           │   └──────────────┘            │
///           │                               │
///           │   ┌──────────────┐            │
///           └──►│ Plant Model  ├──► ŷ_del   │
///               │  (+ delay)   │            │
///               └──────────────┘            │
///                                           │
///   compensated = y_actual - ŷ_del + ŷ_inst ◄┘
/// ```
pub struct SmithPredictor {
    config: SmithPredictorConfig,
    /// Plant model for instantaneous (delay-free) prediction
    model_instant: PlantModel,
    /// Plant model for delayed prediction (runs through delay buffer)
    model_delayed: PlantModel,
    /// Ring buffer of delayed model predictions, indexed by discrete time steps
    delay_buffer: VecDeque<f64>,
    /// Number of discrete time steps corresponding to the current dead time
    delay_steps: usize,
    /// EMA of model error magnitude
    ema_error: f64,
    /// EMA of model output magnitude (for normalised mismatch detection)
    ema_magnitude: f64,
    /// Whether EMA has been initialized
    ema_initialized: bool,
    /// Adaptive dead-time estimate (ms)
    adaptive_dead_time_ms: f64,
    /// Whether adaptive dead time has been initialized
    adaptive_dt_initialized: bool,
    /// Last plant output observed (for dead-time estimation)
    last_plant_output: Option<f64>,
    /// Last control action timestamp (for dead-time estimation)
    last_action_timestamp_ms: Option<f64>,
    /// Recent prediction records for windowed analysis
    recent: VecDeque<PredictionRecord>,
    /// Running statistics
    stats: SmithPredictorStats,
    /// Current control input being applied
    current_input: f64,
    /// Most recent instantaneous model prediction
    latest_instant_prediction: f64,
    /// Most recent delayed model prediction
    latest_delayed_prediction: f64,
    /// Most recent plant output
    latest_plant_output: f64,
    /// Whether in mismatch state
    in_mismatch: bool,
}

impl Default for SmithPredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl SmithPredictor {
    /// Create a new instance with default configuration
    pub fn new() -> Self {
        Self::with_config(SmithPredictorConfig::default())
    }

    /// Create a new instance with the given configuration
    pub fn with_config(config: SmithPredictorConfig) -> Self {
        let delay_steps = if config.dt_ms > 0.0 {
            (config.dead_time_ms / config.dt_ms).round() as usize
        } else {
            0
        };

        let model_instant = PlantModel::new(config.plant_gain, config.plant_time_constant_ms);
        let model_delayed = PlantModel::new(config.plant_gain, config.plant_time_constant_ms);

        let mut delay_buffer = VecDeque::with_capacity(config.delay_buffer_size);
        // Pre-fill delay buffer with zeros
        for _ in 0..delay_steps {
            delay_buffer.push_back(0.0);
        }

        let mut stats = SmithPredictorStats::default();
        stats.effective_dead_time_ms = config.dead_time_ms;

        Self {
            adaptive_dead_time_ms: config.dead_time_ms,
            adaptive_dt_initialized: false,
            model_instant,
            model_delayed,
            delay_buffer,
            delay_steps,
            ema_error: 0.0,
            ema_magnitude: 0.0,
            ema_initialized: false,
            last_plant_output: None,
            last_action_timestamp_ms: None,
            recent: VecDeque::new(),
            stats,
            current_input: 0.0,
            latest_instant_prediction: 0.0,
            latest_delayed_prediction: 0.0,
            latest_plant_output: 0.0,
            in_mismatch: false,
            config,
        }
    }

    /// Main processing function — validates configuration
    pub fn process(&self) -> Result<()> {
        if self.config.dead_time_ms < 0.0 {
            return Err(Error::InvalidInput("dead_time_ms must be >= 0".into()));
        }
        if self.config.plant_time_constant_ms < 0.0 {
            return Err(Error::InvalidInput(
                "plant_time_constant_ms must be >= 0".into(),
            ));
        }
        if self.config.dt_ms <= 0.0 {
            return Err(Error::InvalidInput("dt_ms must be > 0".into()));
        }
        if self.config.ema_decay <= 0.0 || self.config.ema_decay >= 1.0 {
            return Err(Error::InvalidInput("ema_decay must be in (0, 1)".into()));
        }
        if self.config.delay_buffer_size == 0 {
            return Err(Error::InvalidInput("delay_buffer_size must be > 0".into()));
        }
        if self.config.mismatch_threshold <= 0.0 {
            return Err(Error::InvalidInput("mismatch_threshold must be > 0".into()));
        }
        if self.config.latency_ema_decay <= 0.0 || self.config.latency_ema_decay >= 1.0 {
            return Err(Error::InvalidInput(
                "latency_ema_decay must be in (0, 1)".into(),
            ));
        }
        Ok(())
    }

    /// Submit a control action to the predictor
    ///
    /// This should be called each time the controller issues a new control
    /// signal. The predictor steps its internal plant models and updates
    /// the delay buffer.
    pub fn apply_action(&mut self, action: &ControlAction) -> Result<()> {
        self.current_input = action.signal;
        self.last_action_timestamp_ms = Some(action.timestamp_ms);
        self.stats.total_actions += 1;

        // Step the instantaneous model
        self.latest_instant_prediction = self.model_instant.step(action.signal, self.config.dt_ms);

        // Step the delayed model and push into delay buffer
        let delayed_model_output = self.model_delayed.step(action.signal, self.config.dt_ms);
        self.delay_buffer.push_back(delayed_model_output);

        // Keep delay buffer bounded
        while self.delay_buffer.len() > self.config.delay_buffer_size {
            self.delay_buffer.pop_front();
        }

        // The delayed prediction is the value from `delay_steps` ago
        self.latest_delayed_prediction = if self.delay_buffer.len() > self.delay_steps {
            let idx = self.delay_buffer.len() - 1 - self.delay_steps;
            self.delay_buffer[idx]
        } else if !self.delay_buffer.is_empty() {
            self.delay_buffer[0]
        } else {
            0.0
        };

        Ok(())
    }

    /// Provide an observed plant output and get the compensated feedback signal
    ///
    /// The compensated signal removes the effect of dead time:
    ///   compensated = y_actual - ŷ_delayed + ŷ_instantaneous
    pub fn compensate(&mut self, output: &PlantOutput) -> CompensatedOutput {
        self.latest_plant_output = output.value;
        self.last_plant_output = Some(output.value);
        self.stats.total_outputs += 1;

        // Model error: difference between actual and delayed prediction
        let model_error = output.value - self.latest_delayed_prediction;

        // Compensated signal: removes dead time from the loop
        let compensated_signal =
            output.value - self.latest_delayed_prediction + self.latest_instant_prediction;

        // Update EMA of error and magnitude
        let abs_error = model_error.abs();
        let magnitude = output.value.abs().max(self.latest_delayed_prediction.abs());

        if self.ema_initialized {
            self.ema_error =
                self.config.ema_decay * self.ema_error + (1.0 - self.config.ema_decay) * abs_error;
            self.ema_magnitude = self.config.ema_decay * self.ema_magnitude
                + (1.0 - self.config.ema_decay) * magnitude;
        } else {
            self.ema_error = abs_error;
            self.ema_magnitude = magnitude;
            self.ema_initialized = true;
        }

        // Mismatch detection
        let was_mismatch = self.in_mismatch;
        self.in_mismatch = self.detect_mismatch();
        if self.in_mismatch && !was_mismatch {
            self.stats.mismatch_events += 1;
        }

        // Update stats
        self.stats.total_compensations += 1;
        self.stats.sum_sq_error += model_error * model_error;
        self.stats.sum_abs_error += abs_error;
        self.stats.sum_signed_error += model_error;
        if abs_error > self.stats.max_abs_error {
            self.stats.max_abs_error = abs_error;
        }
        if abs_error > self.stats.peak_error {
            self.stats.peak_error = abs_error;
        }

        // Record for windowed analysis
        let record = PredictionRecord {
            predicted: self.latest_delayed_prediction,
            actual: output.value,
            sq_error: model_error * model_error,
        };
        self.recent.push_back(record);
        while self.recent.len() > self.config.window_size {
            self.recent.pop_front();
        }

        CompensatedOutput {
            compensated_signal,
            model_prediction: self.latest_instant_prediction,
            delayed_prediction: self.latest_delayed_prediction,
            plant_output: output.value,
            model_error,
            mismatch_detected: self.in_mismatch,
            effective_dead_time_ms: self.adaptive_dead_time_ms,
        }
    }

    /// Update the dead-time estimate from an observed round-trip latency
    ///
    /// Call this when you measure an actual order round-trip time. The
    /// predictor will adapt its internal dead-time model via EMA.
    pub fn update_dead_time(&mut self, observed_latency_ms: f64) -> Result<()> {
        if observed_latency_ms < 0.0 {
            return Err(Error::InvalidInput(
                "observed_latency_ms must be >= 0".into(),
            ));
        }

        if self.adaptive_dt_initialized {
            self.adaptive_dead_time_ms = self.config.latency_ema_decay * self.adaptive_dead_time_ms
                + (1.0 - self.config.latency_ema_decay) * observed_latency_ms;
        } else {
            self.adaptive_dead_time_ms = observed_latency_ms;
            self.adaptive_dt_initialized = true;
        }

        // Recompute delay steps
        if self.config.dt_ms > 0.0 {
            self.delay_steps = (self.adaptive_dead_time_ms / self.config.dt_ms).round() as usize;
        }

        self.stats.effective_dead_time_ms = self.adaptive_dead_time_ms;
        self.stats.latency_updates += 1;

        Ok(())
    }

    /// Whether a model mismatch is currently detected
    fn detect_mismatch(&self) -> bool {
        if self.stats.total_compensations < self.config.min_samples {
            return false;
        }
        if self.ema_magnitude < 1e-10 {
            // Near-zero magnitude: use absolute threshold
            return self.ema_error > self.config.mismatch_threshold;
        }
        let normalised_error = self.ema_error / self.ema_magnitude;
        normalised_error > self.config.mismatch_threshold
    }

    /// Whether a model mismatch is currently detected (public accessor)
    pub fn is_mismatch_detected(&self) -> bool {
        self.in_mismatch
    }

    /// Current EMA-smoothed model error
    pub fn smoothed_error(&self) -> f64 {
        if self.ema_initialized {
            self.ema_error
        } else {
            0.0
        }
    }

    /// Current effective dead time (ms)
    pub fn effective_dead_time_ms(&self) -> f64 {
        self.adaptive_dead_time_ms
    }

    /// Current delay steps
    pub fn delay_steps(&self) -> usize {
        self.delay_steps
    }

    /// Whether the adaptive dead-time estimate has been initialized
    pub fn is_dead_time_adapted(&self) -> bool {
        self.adaptive_dt_initialized
    }

    /// Latest instantaneous model prediction
    pub fn instant_prediction(&self) -> f64 {
        self.latest_instant_prediction
    }

    /// Latest delayed model prediction
    pub fn delayed_prediction(&self) -> f64 {
        self.latest_delayed_prediction
    }

    /// Reference to running statistics
    pub fn stats(&self) -> &SmithPredictorStats {
        &self.stats
    }

    /// Windowed RMSE
    pub fn windowed_rmse(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let mse: f64 =
            self.recent.iter().map(|r| r.sq_error).sum::<f64>() / self.recent.len() as f64;
        mse.sqrt()
    }

    /// Windowed MAE
    pub fn windowed_mae(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let mae: f64 = self
            .recent
            .iter()
            .map(|r| (r.actual - r.predicted).abs())
            .sum::<f64>()
            / self.recent.len() as f64;
        mae
    }

    /// Check if model quality is degrading (second half of window worse than first)
    pub fn is_degrading(&self) -> bool {
        let n = self.recent.len();
        if n < 10 {
            return false;
        }
        let mid = n / 2;

        let first_half_mse: f64 = self
            .recent
            .iter()
            .take(mid)
            .map(|r| r.sq_error)
            .sum::<f64>()
            / mid as f64;
        let second_half_mse: f64 = self
            .recent
            .iter()
            .skip(mid)
            .map(|r| r.sq_error)
            .sum::<f64>()
            / (n - mid) as f64;

        second_half_mse > first_half_mse * 1.3
    }

    /// Compute confidence in the predictor
    pub fn confidence(&self) -> f64 {
        if self.stats.total_compensations == 0 {
            return 0.0;
        }

        // Sample confidence
        let sample_conf = (self.stats.total_compensations as f64
            / (self.config.min_samples as f64 * 3.0))
            .min(1.0);

        // Model accuracy confidence (based on normalised error)
        let accuracy_conf = if self.ema_magnitude > 1e-10 && self.ema_initialized {
            let norm_err = self.ema_error / self.ema_magnitude;
            (1.0 - norm_err / self.config.mismatch_threshold)
                .max(0.0)
                .min(1.0)
        } else {
            0.5
        };

        (sample_conf * accuracy_conf).sqrt()
    }

    /// Reset all state and statistics
    pub fn reset(&mut self) {
        self.model_instant.reset();
        self.model_delayed.reset();
        self.delay_buffer.clear();

        // Re-fill delay buffer with zeros
        for _ in 0..self.delay_steps {
            self.delay_buffer.push_back(0.0);
        }

        self.ema_error = 0.0;
        self.ema_magnitude = 0.0;
        self.ema_initialized = false;
        self.adaptive_dead_time_ms = self.config.dead_time_ms;
        self.adaptive_dt_initialized = false;
        self.last_plant_output = None;
        self.last_action_timestamp_ms = None;
        self.recent.clear();
        self.current_input = 0.0;
        self.latest_instant_prediction = 0.0;
        self.latest_delayed_prediction = 0.0;
        self.latest_plant_output = 0.0;
        self.in_mismatch = false;

        // Recompute delay steps from config
        if self.config.dt_ms > 0.0 {
            self.delay_steps = (self.config.dead_time_ms / self.config.dt_ms).round() as usize;
        }

        self.stats = SmithPredictorStats::default();
        self.stats.effective_dead_time_ms = self.config.dead_time_ms;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn action(signal: f64, ts: f64) -> ControlAction {
        ControlAction {
            signal,
            timestamp_ms: ts,
        }
    }

    fn output(value: f64, ts: f64) -> PlantOutput {
        PlantOutput {
            value,
            timestamp_ms: ts,
        }
    }

    fn small_config() -> SmithPredictorConfig {
        SmithPredictorConfig {
            dead_time_ms: 10.0,
            plant_gain: 1.0,
            plant_time_constant_ms: 50.0,
            ema_decay: 0.9,
            delay_buffer_size: 100,
            dt_ms: 1.0,
            mismatch_threshold: 0.3,
            min_samples: 5,
            window_size: 100,
            latency_ema_decay: 0.9,
        }
    }

    #[test]
    fn test_basic() {
        let instance = SmithPredictor::new();
        assert!(instance.process().is_ok());
    }

    #[test]
    fn test_default_initial_state() {
        let sp = SmithPredictor::new();
        assert_eq!(sp.instant_prediction(), 0.0);
        assert_eq!(sp.delayed_prediction(), 0.0);
        assert_eq!(sp.smoothed_error(), 0.0);
        assert!(!sp.is_mismatch_detected());
        assert!(!sp.is_dead_time_adapted());
        assert_eq!(sp.stats().total_actions, 0);
        assert_eq!(sp.stats().total_outputs, 0);
    }

    #[test]
    fn test_delay_steps_calculation() {
        let sp = SmithPredictor::with_config(SmithPredictorConfig {
            dead_time_ms: 50.0,
            dt_ms: 5.0,
            ..Default::default()
        });
        assert_eq!(sp.delay_steps(), 10);
    }

    #[test]
    fn test_delay_steps_rounding() {
        let sp = SmithPredictor::with_config(SmithPredictorConfig {
            dead_time_ms: 7.0,
            dt_ms: 3.0,
            ..Default::default()
        });
        // 7/3 = 2.33 → rounds to 2
        assert_eq!(sp.delay_steps(), 2);
    }

    #[test]
    fn test_apply_action_increments_count() {
        let mut sp = SmithPredictor::with_config(small_config());
        sp.apply_action(&action(1.0, 0.0)).unwrap();
        assert_eq!(sp.stats().total_actions, 1);
        sp.apply_action(&action(2.0, 1.0)).unwrap();
        assert_eq!(sp.stats().total_actions, 2);
    }

    #[test]
    fn test_instant_prediction_responds() {
        let mut sp = SmithPredictor::with_config(SmithPredictorConfig {
            plant_gain: 2.0,
            plant_time_constant_ms: 0.0, // instantaneous
            ..small_config()
        });

        sp.apply_action(&action(5.0, 0.0)).unwrap();
        assert!(
            (sp.instant_prediction() - 10.0).abs() < 1e-6,
            "gain=2, input=5 → prediction should be 10, got {}",
            sp.instant_prediction()
        );
    }

    #[test]
    fn test_first_order_model_converges() {
        let mut sp = SmithPredictor::with_config(SmithPredictorConfig {
            plant_gain: 1.0,
            plant_time_constant_ms: 10.0,
            dt_ms: 1.0,
            dead_time_ms: 0.0,
            ..small_config()
        });

        // Apply constant input and step many times
        for i in 0..200 {
            sp.apply_action(&action(1.0, i as f64)).unwrap();
        }

        // Should converge to gain * input = 1.0
        assert!(
            (sp.instant_prediction() - 1.0).abs() < 0.01,
            "first-order model should converge to steady state, got {}",
            sp.instant_prediction()
        );
    }

    #[test]
    fn test_compensated_signal_removes_delay() {
        let mut sp = SmithPredictor::with_config(SmithPredictorConfig {
            dead_time_ms: 5.0,
            plant_gain: 1.0,
            plant_time_constant_ms: 0.0, // instantaneous for simplicity
            dt_ms: 1.0,
            ..small_config()
        });

        // Apply actions to build up the model
        for i in 0..20 {
            sp.apply_action(&action(1.0, i as f64)).unwrap();
        }

        // If the plant output matches the delayed model prediction exactly,
        // the compensated signal should equal the instantaneous prediction
        let delayed = sp.delayed_prediction();
        let comp = sp.compensate(&output(delayed, 20.0));

        assert!(
            (comp.compensated_signal - sp.instant_prediction()).abs() < 1e-6,
            "perfect model → compensated should equal instant prediction: {} vs {}",
            comp.compensated_signal,
            sp.instant_prediction()
        );
        assert!(
            comp.model_error.abs() < 1e-6,
            "perfect model should have ~0 error, got {}",
            comp.model_error
        );
    }

    #[test]
    fn test_model_error_nonzero_on_mismatch() {
        let mut sp = SmithPredictor::with_config(small_config());

        for i in 0..20 {
            sp.apply_action(&action(1.0, i as f64)).unwrap();
        }

        // Feed a plant output that doesn't match the model
        let comp = sp.compensate(&output(999.0, 20.0));
        assert!(
            comp.model_error.abs() > 1.0,
            "model error should be large with mismatched output"
        );
    }

    #[test]
    fn test_compensate_increments_stats() {
        let mut sp = SmithPredictor::with_config(small_config());
        sp.apply_action(&action(1.0, 0.0)).unwrap();
        sp.compensate(&output(1.0, 1.0));

        assert_eq!(sp.stats().total_compensations, 1);
        assert_eq!(sp.stats().total_outputs, 1);
    }

    #[test]
    fn test_mismatch_detection_not_triggered_early() {
        let mut sp = SmithPredictor::with_config(SmithPredictorConfig {
            min_samples: 10,
            ..small_config()
        });

        // Feed wildly wrong outputs but below min_samples
        for i in 0..9 {
            sp.apply_action(&action(1.0, i as f64)).unwrap();
            sp.compensate(&output(1000.0, i as f64 + 0.5));
        }

        assert!(!sp.is_mismatch_detected());
    }

    #[test]
    fn test_mismatch_detection_triggered() {
        let mut sp = SmithPredictor::with_config(SmithPredictorConfig {
            min_samples: 5,
            mismatch_threshold: 0.3,
            ema_decay: 0.5,
            ..small_config()
        });

        // Consistently large model errors
        for i in 0..30 {
            sp.apply_action(&action(1.0, i as f64)).unwrap();
            sp.compensate(&output(1000.0, i as f64 + 0.5));
        }

        assert!(
            sp.is_mismatch_detected(),
            "should detect mismatch with consistently wrong predictions"
        );
        assert!(sp.stats().mismatch_events > 0);
    }

    #[test]
    fn test_no_mismatch_with_perfect_model() {
        let mut sp = SmithPredictor::with_config(SmithPredictorConfig {
            min_samples: 3,
            plant_gain: 1.0,
            plant_time_constant_ms: 0.0,
            dead_time_ms: 0.0,
            ..small_config()
        });

        for i in 0..30 {
            sp.apply_action(&action(1.0, i as f64)).unwrap();
            let pred = sp.delayed_prediction();
            sp.compensate(&output(pred, i as f64 + 0.5));
        }

        assert!(!sp.is_mismatch_detected());
    }

    #[test]
    fn test_update_dead_time() {
        let mut sp = SmithPredictor::with_config(SmithPredictorConfig {
            dead_time_ms: 50.0,
            dt_ms: 1.0,
            latency_ema_decay: 0.5,
            ..small_config()
        });

        assert_eq!(sp.effective_dead_time_ms(), 50.0);

        // First update: initialises directly
        sp.update_dead_time(100.0).unwrap();
        assert!(sp.is_dead_time_adapted());
        assert!((sp.effective_dead_time_ms() - 100.0).abs() < 1e-6);

        // Second update: EMA blends
        sp.update_dead_time(50.0).unwrap();
        // EMA(0.5): 0.5 * 100 + 0.5 * 50 = 75
        assert!(
            (sp.effective_dead_time_ms() - 75.0).abs() < 1e-6,
            "expected 75, got {}",
            sp.effective_dead_time_ms()
        );

        assert_eq!(sp.stats().latency_updates, 2);
    }

    #[test]
    fn test_update_dead_time_negative_rejected() {
        let mut sp = SmithPredictor::new();
        assert!(sp.update_dead_time(-1.0).is_err());
    }

    #[test]
    fn test_delay_steps_update_on_dead_time_change() {
        let mut sp = SmithPredictor::with_config(SmithPredictorConfig {
            dead_time_ms: 10.0,
            dt_ms: 1.0,
            ..small_config()
        });

        assert_eq!(sp.delay_steps(), 10);

        sp.update_dead_time(20.0).unwrap();
        assert_eq!(sp.delay_steps(), 20);
    }

    #[test]
    fn test_stats_error_tracking() {
        let mut sp = SmithPredictor::with_config(small_config());

        sp.apply_action(&action(1.0, 0.0)).unwrap();
        let comp = sp.compensate(&output(5.0, 1.0));

        assert!(sp.stats().sum_sq_error > 0.0);
        assert!(sp.stats().sum_abs_error > 0.0);
        assert!((sp.stats().max_abs_error - comp.model_error.abs()).abs() < 1e-10);
    }

    #[test]
    fn test_stats_mse_and_rmse() {
        let mut sp = SmithPredictor::with_config(SmithPredictorConfig {
            plant_gain: 1.0,
            plant_time_constant_ms: 0.0,
            dead_time_ms: 0.0,
            ..small_config()
        });

        // Perfect predictions → MSE should be ~0
        for i in 0..20 {
            sp.apply_action(&action(1.0, i as f64)).unwrap();
            let pred = sp.delayed_prediction();
            sp.compensate(&output(pred, i as f64 + 0.5));
        }

        assert!(
            sp.stats().mse() < 1e-10,
            "perfect model should have ~0 MSE, got {}",
            sp.stats().mse()
        );
        assert!(sp.stats().rmse() < 1e-5);
    }

    #[test]
    fn test_stats_bias() {
        let mut sp = SmithPredictor::with_config(SmithPredictorConfig {
            plant_gain: 1.0,
            plant_time_constant_ms: 0.0,
            dead_time_ms: 0.0,
            ..small_config()
        });

        // Consistently overshoot: actual is always higher than predicted
        for i in 0..20 {
            sp.apply_action(&action(1.0, i as f64)).unwrap();
            let pred = sp.delayed_prediction();
            sp.compensate(&output(pred + 10.0, i as f64 + 0.5));
        }

        assert!(
            sp.stats().bias() > 0.0,
            "positive error should produce positive bias, got {}",
            sp.stats().bias()
        );
    }

    #[test]
    fn test_stats_defaults() {
        let stats = SmithPredictorStats::default();
        assert_eq!(stats.mse(), 0.0);
        assert_eq!(stats.rmse(), 0.0);
        assert_eq!(stats.mae(), 0.0);
        assert_eq!(stats.bias(), 0.0);
    }

    #[test]
    fn test_windowed_rmse() {
        let mut sp = SmithPredictor::with_config(SmithPredictorConfig {
            window_size: 10,
            plant_gain: 1.0,
            plant_time_constant_ms: 0.0,
            dead_time_ms: 0.0,
            ..small_config()
        });

        for i in 0..10 {
            sp.apply_action(&action(1.0, i as f64)).unwrap();
            let pred = sp.delayed_prediction();
            sp.compensate(&output(pred, i as f64 + 0.5));
        }

        assert!(
            sp.windowed_rmse() < 1e-5,
            "perfect predictions should yield ~0 windowed RMSE"
        );
    }

    #[test]
    fn test_windowed_mae() {
        let mut sp = SmithPredictor::with_config(SmithPredictorConfig {
            window_size: 10,
            ..small_config()
        });

        sp.apply_action(&action(1.0, 0.0)).unwrap();
        let pred = sp.delayed_prediction();
        sp.compensate(&output(pred + 2.0, 1.0));

        assert!(
            (sp.windowed_mae() - 2.0).abs() < 1e-6,
            "single error of 2.0 should give MAE of 2.0, got {}",
            sp.windowed_mae()
        );
    }

    #[test]
    fn test_is_degrading() {
        let mut sp = SmithPredictor::with_config(SmithPredictorConfig {
            window_size: 20,
            plant_gain: 1.0,
            plant_time_constant_ms: 0.0,
            dead_time_ms: 0.0,
            ..small_config()
        });

        // First half: perfect predictions
        for i in 0..10 {
            sp.apply_action(&action(1.0, i as f64)).unwrap();
            let pred = sp.delayed_prediction();
            sp.compensate(&output(pred, i as f64 + 0.5));
        }

        // Second half: terrible predictions
        for i in 10..20 {
            sp.apply_action(&action(1.0, i as f64)).unwrap();
            sp.compensate(&output(999.0, i as f64 + 0.5));
        }

        assert!(sp.is_degrading());
    }

    #[test]
    fn test_not_degrading_consistent() {
        let mut sp = SmithPredictor::with_config(SmithPredictorConfig {
            window_size: 20,
            ..small_config()
        });

        for i in 0..20 {
            sp.apply_action(&action(1.0, i as f64)).unwrap();
            sp.compensate(&output(5.0, i as f64 + 0.5));
        }

        assert!(!sp.is_degrading());
    }

    #[test]
    fn test_not_degrading_insufficient_data() {
        let mut sp = SmithPredictor::new();
        for i in 0..4 {
            sp.apply_action(&action(1.0, i as f64)).unwrap();
            sp.compensate(&output(999.0, i as f64 + 0.5));
        }
        assert!(!sp.is_degrading());
    }

    #[test]
    fn test_confidence_zero_without_data() {
        let sp = SmithPredictor::new();
        assert!((sp.confidence() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_confidence_increases_with_data() {
        let mut sp = SmithPredictor::with_config(SmithPredictorConfig {
            min_samples: 5,
            plant_gain: 1.0,
            plant_time_constant_ms: 0.0,
            dead_time_ms: 0.0,
            ..small_config()
        });

        let conf0 = sp.confidence();

        for i in 0..30 {
            sp.apply_action(&action(1.0, i as f64)).unwrap();
            let pred = sp.delayed_prediction();
            sp.compensate(&output(pred, i as f64 + 0.5));
        }

        let conf1 = sp.confidence();
        assert!(
            conf1 > conf0,
            "confidence should increase with accurate data: {} vs {}",
            conf1,
            conf0
        );
    }

    #[test]
    fn test_reset() {
        let mut sp = SmithPredictor::with_config(small_config());

        for i in 0..20 {
            sp.apply_action(&action(1.0, i as f64)).unwrap();
            sp.compensate(&output(5.0, i as f64 + 0.5));
        }
        sp.update_dead_time(100.0).unwrap();

        assert!(sp.stats().total_actions > 0);
        assert!(sp.is_dead_time_adapted());

        sp.reset();

        assert_eq!(sp.stats().total_actions, 0);
        assert_eq!(sp.stats().total_compensations, 0);
        assert!(!sp.is_dead_time_adapted());
        assert!(!sp.is_mismatch_detected());
        assert_eq!(sp.instant_prediction(), 0.0);
        assert_eq!(sp.delayed_prediction(), 0.0);
        assert_eq!(sp.smoothed_error(), 0.0);
        assert_eq!(sp.effective_dead_time_ms(), sp.config.dead_time_ms);
    }

    #[test]
    fn test_window_eviction() {
        let mut sp = SmithPredictor::with_config(SmithPredictorConfig {
            window_size: 5,
            ..small_config()
        });

        for i in 0..20 {
            sp.apply_action(&action(1.0, i as f64)).unwrap();
            sp.compensate(&output(1.0, i as f64 + 0.5));
        }

        assert_eq!(sp.recent.len(), 5);
        assert_eq!(sp.stats().total_compensations, 20);
    }

    #[test]
    fn test_invalid_config_negative_dead_time() {
        let sp = SmithPredictor::with_config(SmithPredictorConfig {
            dead_time_ms: -1.0,
            ..Default::default()
        });
        assert!(sp.process().is_err());
    }

    #[test]
    fn test_invalid_config_negative_time_constant() {
        let sp = SmithPredictor::with_config(SmithPredictorConfig {
            plant_time_constant_ms: -1.0,
            ..Default::default()
        });
        assert!(sp.process().is_err());
    }

    #[test]
    fn test_invalid_config_zero_dt() {
        let sp = SmithPredictor::with_config(SmithPredictorConfig {
            dt_ms: 0.0,
            ..Default::default()
        });
        assert!(sp.process().is_err());
    }

    #[test]
    fn test_invalid_config_bad_ema_decay() {
        let sp = SmithPredictor::with_config(SmithPredictorConfig {
            ema_decay: 1.0,
            ..Default::default()
        });
        assert!(sp.process().is_err());
    }

    #[test]
    fn test_invalid_config_zero_buffer() {
        let sp = SmithPredictor::with_config(SmithPredictorConfig {
            delay_buffer_size: 0,
            ..Default::default()
        });
        assert!(sp.process().is_err());
    }

    #[test]
    fn test_invalid_config_zero_mismatch_threshold() {
        let sp = SmithPredictor::with_config(SmithPredictorConfig {
            mismatch_threshold: 0.0,
            ..Default::default()
        });
        assert!(sp.process().is_err());
    }

    #[test]
    fn test_invalid_config_bad_latency_ema() {
        let sp = SmithPredictor::with_config(SmithPredictorConfig {
            latency_ema_decay: 0.0,
            ..Default::default()
        });
        assert!(sp.process().is_err());
    }

    #[test]
    fn test_zero_dead_time() {
        let mut sp = SmithPredictor::with_config(SmithPredictorConfig {
            dead_time_ms: 0.0,
            plant_gain: 1.0,
            plant_time_constant_ms: 0.0,
            dt_ms: 1.0,
            ..small_config()
        });

        assert_eq!(sp.delay_steps(), 0);

        sp.apply_action(&action(3.0, 0.0)).unwrap();

        // With zero dead time and instantaneous model, delayed = instant
        assert!(
            (sp.instant_prediction() - sp.delayed_prediction()).abs() < 1e-6,
            "zero dead time: instant={} should ≈ delayed={}",
            sp.instant_prediction(),
            sp.delayed_prediction()
        );
    }

    #[test]
    fn test_compensated_output_structure() {
        let mut sp = SmithPredictor::with_config(small_config());

        sp.apply_action(&action(1.0, 0.0)).unwrap();
        let comp = sp.compensate(&output(2.0, 1.0));

        assert_eq!(comp.plant_output, 2.0);
        assert!(comp.effective_dead_time_ms > 0.0);
        // compensated = plant_output - delayed + instant
        let expected = comp.plant_output - comp.delayed_prediction + comp.model_prediction;
        assert!(
            (comp.compensated_signal - expected).abs() < 1e-10,
            "compensated formula: {} vs expected {}",
            comp.compensated_signal,
            expected
        );
    }

    #[test]
    fn test_delayed_prediction_lags_instant() {
        let mut sp = SmithPredictor::with_config(SmithPredictorConfig {
            dead_time_ms: 5.0,
            plant_gain: 1.0,
            plant_time_constant_ms: 10.0,
            dt_ms: 1.0,
            ..small_config()
        });

        // Apply step input
        for i in 0..3 {
            sp.apply_action(&action(1.0, i as f64)).unwrap();
        }

        // The delayed prediction should be behind the instant prediction
        // since the delay buffer introduces a lag
        // (instant model has been stepping for 3 steps, delayed is looking 5 steps back)
        assert!(
            sp.instant_prediction() >= sp.delayed_prediction(),
            "instant={} should >= delayed={} during ramp-up",
            sp.instant_prediction(),
            sp.delayed_prediction()
        );
    }

    #[test]
    fn test_windowed_empty() {
        let sp = SmithPredictor::new();
        assert_eq!(sp.windowed_rmse(), 0.0);
        assert_eq!(sp.windowed_mae(), 0.0);
    }

    #[test]
    fn test_plant_model_step_zero_time_constant() {
        let mut model = PlantModel::new(2.0, 0.0);
        let out = model.step(3.0, 1.0);
        assert!((out - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_plant_model_reset() {
        let mut model = PlantModel::new(1.0, 10.0);
        model.step(5.0, 10.0);
        assert!(model.output.abs() > 0.0);
        model.reset();
        assert_eq!(model.output, 0.0);
    }

    #[test]
    fn test_multiple_actions_before_compensate() {
        let mut sp = SmithPredictor::with_config(small_config());

        // Many actions without any compensation step
        for i in 0..50 {
            sp.apply_action(&action(1.0, i as f64)).unwrap();
        }

        assert_eq!(sp.stats().total_actions, 50);
        assert_eq!(sp.stats().total_compensations, 0);

        // Then compensate
        let comp = sp.compensate(&output(1.0, 50.0));
        assert_eq!(sp.stats().total_compensations, 1);
        assert!(comp.compensated_signal.is_finite());
    }
}
