//! Adaptive correction — online learning from execution errors
//!
//! Part of the Cerebellum region
//! Component: error_correction
//!
//! Learns from observed execution error patterns and adapts correction
//! parameters online using gradient-based gain adaptation. Detects
//! error regimes (trending, mean-reverting, volatile) and applies
//! regime-specific correction strategies with cooldown-gated updates
//! to prevent over-adaptation.
//!
//! Key features:
//! - Multi-parameter adaptive gain adjustment via gradient descent
//! - Error regime detection (trending, mean-reverting, volatile, stable)
//! - Cooldown-gated updates to prevent oscillatory over-correction
//! - Learning rate scheduling with decay and warm-up
//! - Momentum-based parameter updates for smoother adaptation
//! - Parameter bounds enforcement with configurable limits
//! - Running statistics and adaptation quality tracking

use crate::common::{Error, Result};
use std::collections::VecDeque;

/// Configuration for the adaptive correction module
#[derive(Debug, Clone)]
pub struct AdaptiveCorrectionConfig {
    /// Number of adaptive parameters (gains) to maintain
    pub num_parameters: usize,
    /// Initial learning rate for gradient updates
    pub learning_rate: f64,
    /// Learning rate decay factor applied per adaptation step (0 < decay <= 1)
    pub learning_rate_decay: f64,
    /// Minimum learning rate floor
    pub min_learning_rate: f64,
    /// Momentum coefficient for parameter updates (0 = no momentum)
    pub momentum: f64,
    /// Cooldown period: minimum steps between parameter updates
    pub cooldown_steps: usize,
    /// EMA decay for error signal smoothing (0 < decay < 1)
    pub ema_decay: f64,
    /// Window size for regime detection
    pub regime_window: usize,
    /// Threshold for classifying error as "trending" (autocorrelation > threshold)
    pub trending_threshold: f64,
    /// Threshold for classifying error as "volatile" (CV > threshold)
    pub volatile_threshold: f64,
    /// Maximum parameter value (upper bound for all parameters)
    pub param_max: f64,
    /// Minimum parameter value (lower bound for all parameters)
    pub param_min: f64,
    /// Maximum number of observations in the sliding window
    pub window_size: usize,
    /// Warm-up steps before adaptation begins
    pub warmup_steps: usize,
    /// Regime-specific gain multiplier for trending errors
    pub trending_multiplier: f64,
    /// Regime-specific gain multiplier for mean-reverting errors
    pub mean_reverting_multiplier: f64,
    /// Regime-specific gain multiplier for volatile errors
    pub volatile_multiplier: f64,
}

impl Default for AdaptiveCorrectionConfig {
    fn default() -> Self {
        Self {
            num_parameters: 3,
            learning_rate: 0.01,
            learning_rate_decay: 0.999,
            min_learning_rate: 0.0001,
            momentum: 0.9,
            cooldown_steps: 5,
            ema_decay: 0.93,
            regime_window: 30,
            trending_threshold: 0.3,
            volatile_threshold: 1.5,
            param_max: 10.0,
            param_min: -10.0,
            window_size: 500,
            warmup_steps: 10,
            trending_multiplier: 1.5,
            mean_reverting_multiplier: 0.5,
            volatile_multiplier: 0.3,
        }
    }
}

/// Detected error regime
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorRegime {
    /// Errors are stable and small — minimal correction needed
    Stable,
    /// Errors show autocorrelation / persistent drift — aggressive correction
    Trending,
    /// Errors oscillate around zero — reduce correction to avoid chasing noise
    MeanReverting,
    /// Errors are large and unpredictable — cautious correction
    Volatile,
}

/// An error observation for the adaptive module
#[derive(Debug, Clone)]
pub struct ErrorObservation {
    /// Error values for each parameter dimension
    pub errors: Vec<f64>,
    /// Optional gradient hint: if provided, used directly instead of estimated
    pub gradient_hint: Option<Vec<f64>>,
}

/// Result of an adaptation step
#[derive(Debug, Clone)]
pub struct AdaptationResult {
    /// Updated parameter values
    pub parameters: Vec<f64>,
    /// Whether parameters were actually updated this step (vs cooldown)
    pub updated: bool,
    /// Current detected error regime
    pub regime: ErrorRegime,
    /// Current effective learning rate
    pub effective_learning_rate: f64,
    /// Regime-specific gain multiplier applied
    pub regime_multiplier: f64,
    /// Parameter deltas (changes from previous values)
    pub deltas: Vec<f64>,
    /// Whether the system is still in warm-up phase
    pub in_warmup: bool,
}

/// Running statistics for the adaptive correction module
#[derive(Debug, Clone, Default)]
pub struct AdaptiveCorrectionStats {
    /// Total observations processed
    pub total_observations: usize,
    /// Total parameter update steps performed
    pub total_updates: usize,
    /// Steps skipped due to cooldown
    pub cooldown_skips: usize,
    /// Steps skipped due to warm-up
    pub warmup_skips: usize,
    /// Sum of squared parameter deltas (for tracking adaptation magnitude)
    pub sum_sq_delta: f64,
    /// Maximum single-step parameter delta
    pub max_delta: f64,
    /// Current learning rate
    pub current_learning_rate: f64,
    /// Count of each regime detected
    pub regime_counts: [usize; 4],
    /// Sum of absolute errors across all observations and dimensions
    pub sum_abs_error: f64,
    /// Sum of squared errors
    pub sum_sq_error: f64,
    /// Maximum absolute error observed
    pub max_abs_error: f64,
    /// Number of times parameters hit bounds
    pub bound_clamp_count: usize,
}

impl AdaptiveCorrectionStats {
    /// Mean absolute error across all observations
    pub fn mean_abs_error(&self) -> f64 {
        if self.total_observations == 0 {
            return 0.0;
        }
        self.sum_abs_error / self.total_observations as f64
    }

    /// Mean squared error
    pub fn mse(&self) -> f64 {
        if self.total_observations == 0 {
            return 0.0;
        }
        self.sum_sq_error / self.total_observations as f64
    }

    /// Root mean squared error
    pub fn rmse(&self) -> f64 {
        self.mse().sqrt()
    }

    /// Mean parameter delta magnitude per update
    pub fn mean_delta_magnitude(&self) -> f64 {
        if self.total_updates == 0 {
            return 0.0;
        }
        (self.sum_sq_delta / self.total_updates as f64).sqrt()
    }

    /// Adaptation rate: fraction of steps that resulted in parameter updates
    pub fn adaptation_rate(&self) -> f64 {
        if self.total_observations == 0 {
            return 0.0;
        }
        self.total_updates as f64 / self.total_observations as f64
    }

    /// Get count for a specific regime
    pub fn regime_count(&self, regime: ErrorRegime) -> usize {
        match regime {
            ErrorRegime::Stable => self.regime_counts[0],
            ErrorRegime::Trending => self.regime_counts[1],
            ErrorRegime::MeanReverting => self.regime_counts[2],
            ErrorRegime::Volatile => self.regime_counts[3],
        }
    }

    /// Dominant regime (most frequently detected)
    pub fn dominant_regime(&self) -> ErrorRegime {
        let max_idx = self
            .regime_counts
            .iter()
            .enumerate()
            .max_by_key(|(_, c)| *c)
            .map(|(i, _)| i)
            .unwrap_or(0);
        match max_idx {
            0 => ErrorRegime::Stable,
            1 => ErrorRegime::Trending,
            2 => ErrorRegime::MeanReverting,
            3 => ErrorRegime::Volatile,
            _ => ErrorRegime::Stable,
        }
    }
}

/// Internal record for windowed analysis
#[derive(Debug, Clone)]
struct ObservationRecord {
    _errors: Vec<f64>,
    _regime: ErrorRegime,
    composite_error: f64,
}

/// Adaptive correction module
///
/// Learns from observed execution error patterns using gradient-based
/// parameter adaptation with regime detection and cooldown gating.
/// Maintains a set of adaptive parameters that can be used by the
/// execution engine to adjust behaviour in real time.
pub struct AdaptiveCorrection {
    config: AdaptiveCorrectionConfig,
    /// Current adaptive parameter values
    parameters: Vec<f64>,
    /// Momentum accumulators (velocity) for each parameter
    velocity: Vec<f64>,
    /// EMA-smoothed error per dimension
    ema_errors: Vec<f64>,
    /// Whether EMA has been initialized
    ema_initialized: bool,
    /// Steps since the last parameter update
    steps_since_update: usize,
    /// Total steps processed (for warm-up tracking)
    total_steps: usize,
    /// Current learning rate (decays over time)
    current_learning_rate: f64,
    /// Current detected regime
    current_regime: ErrorRegime,
    /// Recent observations for regime detection
    recent: VecDeque<ObservationRecord>,
    /// Error history for regime detection (composite errors only)
    error_history: VecDeque<f64>,
    /// Running statistics
    stats: AdaptiveCorrectionStats,
}

impl Default for AdaptiveCorrection {
    fn default() -> Self {
        Self::new()
    }
}

impl AdaptiveCorrection {
    /// Create a new instance with default configuration
    pub fn new() -> Self {
        Self::with_config(AdaptiveCorrectionConfig::default())
    }

    /// Create a new instance with the given configuration
    pub fn with_config(config: AdaptiveCorrectionConfig) -> Self {
        let n = config.num_parameters;
        let lr = config.learning_rate;
        let mut stats = AdaptiveCorrectionStats::default();
        stats.current_learning_rate = lr;

        Self {
            parameters: vec![0.0; n],
            velocity: vec![0.0; n],
            ema_errors: vec![0.0; n],
            ema_initialized: false,
            steps_since_update: 0,
            total_steps: 0,
            current_learning_rate: lr,
            current_regime: ErrorRegime::Stable,
            recent: VecDeque::new(),
            error_history: VecDeque::new(),
            stats,
            config,
        }
    }

    /// Main processing function — validates configuration
    pub fn process(&self) -> Result<()> {
        if self.config.num_parameters == 0 {
            return Err(Error::InvalidInput("num_parameters must be > 0".into()));
        }
        if self.config.learning_rate <= 0.0 {
            return Err(Error::InvalidInput("learning_rate must be > 0".into()));
        }
        if self.config.learning_rate_decay <= 0.0 || self.config.learning_rate_decay > 1.0 {
            return Err(Error::InvalidInput(
                "learning_rate_decay must be in (0, 1]".into(),
            ));
        }
        if self.config.min_learning_rate < 0.0 {
            return Err(Error::InvalidInput("min_learning_rate must be >= 0".into()));
        }
        if self.config.momentum < 0.0 || self.config.momentum >= 1.0 {
            return Err(Error::InvalidInput("momentum must be in [0, 1)".into()));
        }
        if self.config.ema_decay <= 0.0 || self.config.ema_decay >= 1.0 {
            return Err(Error::InvalidInput("ema_decay must be in (0, 1)".into()));
        }
        if self.config.param_max <= self.config.param_min {
            return Err(Error::InvalidInput("param_max must be > param_min".into()));
        }
        if self.config.window_size == 0 {
            return Err(Error::InvalidInput("window_size must be > 0".into()));
        }
        if self.config.regime_window == 0 {
            return Err(Error::InvalidInput("regime_window must be > 0".into()));
        }
        if self.config.trending_threshold <= 0.0 || self.config.trending_threshold >= 1.0 {
            return Err(Error::InvalidInput(
                "trending_threshold must be in (0, 1)".into(),
            ));
        }
        if self.config.volatile_threshold <= 0.0 {
            return Err(Error::InvalidInput("volatile_threshold must be > 0".into()));
        }
        Ok(())
    }

    /// Process an error observation and potentially adapt parameters
    ///
    /// Returns the current parameters and adaptation metadata.
    pub fn observe(&mut self, obs: &ErrorObservation) -> Result<AdaptationResult> {
        if obs.errors.len() != self.config.num_parameters {
            return Err(Error::InvalidInput(format!(
                "expected {} errors, got {}",
                self.config.num_parameters,
                obs.errors.len()
            )));
        }
        if let Some(ref hints) = obs.gradient_hint {
            if hints.len() != self.config.num_parameters {
                return Err(Error::InvalidInput(format!(
                    "gradient_hint length {} != num_parameters {}",
                    hints.len(),
                    self.config.num_parameters
                )));
            }
        }

        self.total_steps += 1;
        self.steps_since_update += 1;

        // Compute composite error
        let composite_error: f64 = obs.errors.iter().map(|e| e * e).sum::<f64>().sqrt();

        // Update EMA
        if self.ema_initialized {
            for (ema, err) in self.ema_errors.iter_mut().zip(obs.errors.iter()) {
                *ema = self.config.ema_decay * *ema + (1.0 - self.config.ema_decay) * err;
            }
        } else {
            self.ema_errors = obs.errors.clone();
            self.ema_initialized = true;
        }

        // Update error history for regime detection
        self.error_history.push_back(composite_error);
        while self.error_history.len() > self.config.regime_window {
            self.error_history.pop_front();
        }

        // Detect regime
        self.current_regime = self.detect_regime();

        // Update stats
        self.stats.total_observations += 1;
        let abs_err_sum: f64 = obs.errors.iter().map(|e| e.abs()).sum();
        let sq_err_sum: f64 = obs.errors.iter().map(|e| e * e).sum();
        self.stats.sum_abs_error += abs_err_sum;
        self.stats.sum_sq_error += sq_err_sum;
        if composite_error > self.stats.max_abs_error {
            self.stats.max_abs_error = composite_error;
        }

        // Regime counting
        match self.current_regime {
            ErrorRegime::Stable => self.stats.regime_counts[0] += 1,
            ErrorRegime::Trending => self.stats.regime_counts[1] += 1,
            ErrorRegime::MeanReverting => self.stats.regime_counts[2] += 1,
            ErrorRegime::Volatile => self.stats.regime_counts[3] += 1,
        }

        // Record for windowed analysis
        let record = ObservationRecord {
            _errors: obs.errors.clone(),
            _regime: self.current_regime,
            composite_error,
        };
        self.recent.push_back(record);
        while self.recent.len() > self.config.window_size {
            self.recent.pop_front();
        }

        // Determine regime-specific multiplier
        let regime_multiplier = match self.current_regime {
            ErrorRegime::Stable => 1.0,
            ErrorRegime::Trending => self.config.trending_multiplier,
            ErrorRegime::MeanReverting => self.config.mean_reverting_multiplier,
            ErrorRegime::Volatile => self.config.volatile_multiplier,
        };

        // Check if we should update parameters
        let in_warmup = self.total_steps <= self.config.warmup_steps;
        let cooldown_elapsed = self.steps_since_update >= self.config.cooldown_steps;
        let should_update = !in_warmup && cooldown_elapsed;

        let mut deltas = vec![0.0; self.config.num_parameters];

        if should_update {
            // Compute gradients
            let gradients = if let Some(ref hints) = obs.gradient_hint {
                hints.clone()
            } else {
                // Estimate gradient from smoothed errors:
                // gradient ≈ -error (steepest descent on squared error)
                self.ema_errors.iter().map(|e| -e).collect()
            };

            // Apply momentum and learning rate
            let effective_lr = self.current_learning_rate * regime_multiplier;

            for i in 0..self.config.num_parameters {
                // Momentum update: v = momentum * v + lr * gradient
                self.velocity[i] =
                    self.config.momentum * self.velocity[i] + effective_lr * gradients[i];

                let delta = self.velocity[i];
                deltas[i] = delta;

                // Update parameter
                let new_param = self.parameters[i] + delta;

                // Clamp to bounds
                let clamped = new_param.clamp(self.config.param_min, self.config.param_max);
                if (new_param - clamped).abs() > 1e-10 {
                    self.stats.bound_clamp_count += 1;
                }
                self.parameters[i] = clamped;
            }

            // Decay learning rate
            self.current_learning_rate = (self.current_learning_rate
                * self.config.learning_rate_decay)
                .max(self.config.min_learning_rate);

            // Update stats
            self.stats.total_updates += 1;
            let delta_sq_sum: f64 = deltas.iter().map(|d| d * d).sum();
            self.stats.sum_sq_delta += delta_sq_sum;
            let max_d = deltas.iter().map(|d| d.abs()).fold(0.0f64, |a, b| a.max(b));
            if max_d > self.stats.max_delta {
                self.stats.max_delta = max_d;
            }

            self.steps_since_update = 0;
        } else if in_warmup {
            self.stats.warmup_skips += 1;
        } else {
            self.stats.cooldown_skips += 1;
        }

        self.stats.current_learning_rate = self.current_learning_rate;

        Ok(AdaptationResult {
            parameters: self.parameters.clone(),
            updated: should_update,
            regime: self.current_regime,
            effective_learning_rate: self.current_learning_rate * regime_multiplier,
            regime_multiplier,
            deltas,
            in_warmup,
        })
    }

    /// Detect the current error regime from recent error history
    fn detect_regime(&self) -> ErrorRegime {
        let n = self.error_history.len();
        if n < 4 {
            return ErrorRegime::Stable;
        }

        let values: Vec<f64> = self.error_history.iter().copied().collect();

        // Compute mean and standard deviation
        let mean = values.iter().sum::<f64>() / n as f64;
        let variance = values.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n as f64;
        let std_dev = variance.max(0.0).sqrt();

        // Coefficient of variation
        let cv = if mean.abs() > 1e-10 {
            std_dev / mean.abs()
        } else {
            0.0
        };

        // Check volatile first
        if cv > self.config.volatile_threshold {
            return ErrorRegime::Volatile;
        }

        // Compute lag-1 autocorrelation for trending detection
        let autocorrelation = self.compute_autocorrelation(&values, mean, variance);

        if autocorrelation > self.config.trending_threshold {
            return ErrorRegime::Trending;
        }

        if autocorrelation < -self.config.trending_threshold {
            return ErrorRegime::MeanReverting;
        }

        // Check if errors are small relative to history
        if std_dev < 0.1 && mean.abs() < 0.1 {
            return ErrorRegime::Stable;
        }

        ErrorRegime::Stable
    }

    /// Compute lag-1 autocorrelation of a time series
    fn compute_autocorrelation(&self, values: &[f64], mean: f64, variance: f64) -> f64 {
        if values.len() < 2 || variance < 1e-15 {
            return 0.0;
        }

        let n = values.len();
        let mut covariance = 0.0;
        for i in 1..n {
            covariance += (values[i] - mean) * (values[i - 1] - mean);
        }
        covariance /= (n - 1) as f64;

        (covariance / variance).clamp(-1.0, 1.0)
    }

    /// Get the current adaptive parameters
    pub fn parameters(&self) -> &[f64] {
        &self.parameters
    }

    /// Get a specific parameter by index
    pub fn parameter(&self, index: usize) -> Result<f64> {
        if index >= self.config.num_parameters {
            return Err(Error::InvalidInput(format!(
                "parameter index {} out of range (max {})",
                index,
                self.config.num_parameters - 1
            )));
        }
        Ok(self.parameters[index])
    }

    /// Manually set a parameter value (clamped to bounds)
    pub fn set_parameter(&mut self, index: usize, value: f64) -> Result<()> {
        if index >= self.config.num_parameters {
            return Err(Error::InvalidInput(format!(
                "parameter index {} out of range (max {})",
                index,
                self.config.num_parameters - 1
            )));
        }
        self.parameters[index] = value.clamp(self.config.param_min, self.config.param_max);
        Ok(())
    }

    /// Manually set all parameters at once
    pub fn set_parameters(&mut self, params: &[f64]) -> Result<()> {
        if params.len() != self.config.num_parameters {
            return Err(Error::InvalidInput(format!(
                "expected {} parameters, got {}",
                self.config.num_parameters,
                params.len()
            )));
        }
        for (i, &v) in params.iter().enumerate() {
            self.parameters[i] = v.clamp(self.config.param_min, self.config.param_max);
        }
        Ok(())
    }

    /// Current detected error regime
    pub fn current_regime(&self) -> ErrorRegime {
        self.current_regime
    }

    /// Current effective learning rate (before regime multiplier)
    pub fn current_learning_rate(&self) -> f64 {
        self.current_learning_rate
    }

    /// Current momentum velocity values
    pub fn velocities(&self) -> &[f64] {
        &self.velocity
    }

    /// EMA-smoothed errors per dimension
    pub fn smoothed_errors(&self) -> &[f64] {
        &self.ema_errors
    }

    /// Whether the system is still in warm-up phase
    pub fn in_warmup(&self) -> bool {
        self.total_steps < self.config.warmup_steps
    }

    /// Total steps processed
    pub fn step_count(&self) -> usize {
        self.total_steps
    }

    /// Total parameter updates performed
    pub fn update_count(&self) -> usize {
        self.stats.total_updates
    }

    /// Steps since the last parameter update
    pub fn steps_since_update(&self) -> usize {
        self.steps_since_update
    }

    /// Number of observations in the sliding window
    pub fn window_count(&self) -> usize {
        self.recent.len()
    }

    /// Reference to running statistics
    pub fn stats(&self) -> &AdaptiveCorrectionStats {
        &self.stats
    }

    /// Windowed mean composite error
    pub fn windowed_mean_error(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.recent.iter().map(|r| r.composite_error).sum();
        sum / self.recent.len() as f64
    }

    /// Windowed error standard deviation
    pub fn windowed_error_std(&self) -> f64 {
        let n = self.recent.len();
        if n < 2 {
            return 0.0;
        }
        let mean = self.windowed_mean_error();
        let variance: f64 = self
            .recent
            .iter()
            .map(|r| (r.composite_error - mean) * (r.composite_error - mean))
            .sum::<f64>()
            / n as f64;
        variance.max(0.0).sqrt()
    }

    /// Check if errors are improving (second half of window has lower errors)
    pub fn is_improving(&self) -> bool {
        let n = self.recent.len();
        if n < 10 {
            return false;
        }
        let mid = n / 2;

        let first_half_mean: f64 = self
            .recent
            .iter()
            .take(mid)
            .map(|r| r.composite_error)
            .sum::<f64>()
            / mid as f64;
        let second_half_mean: f64 = self
            .recent
            .iter()
            .skip(mid)
            .map(|r| r.composite_error)
            .sum::<f64>()
            / (n - mid) as f64;

        // Improving if second half is at least 10% lower
        second_half_mean < first_half_mean * 0.9
    }

    /// Check if errors are worsening (second half of window has higher errors)
    pub fn is_worsening(&self) -> bool {
        let n = self.recent.len();
        if n < 10 {
            return false;
        }
        let mid = n / 2;

        let first_half_mean: f64 = self
            .recent
            .iter()
            .take(mid)
            .map(|r| r.composite_error)
            .sum::<f64>()
            / mid as f64;
        let second_half_mean: f64 = self
            .recent
            .iter()
            .skip(mid)
            .map(|r| r.composite_error)
            .sum::<f64>()
            / (n - mid) as f64;

        // Worsening if second half is >20% higher
        second_half_mean > first_half_mean * 1.2
    }

    /// Compute the parameter norm (L2 magnitude)
    pub fn parameter_norm(&self) -> f64 {
        self.parameters.iter().map(|p| p * p).sum::<f64>().sqrt()
    }

    /// Compute the velocity norm (L2 magnitude of momentum)
    pub fn velocity_norm(&self) -> f64 {
        self.velocity.iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    /// Reset all state and statistics
    pub fn reset(&mut self) {
        let n = self.config.num_parameters;
        self.parameters = vec![0.0; n];
        self.velocity = vec![0.0; n];
        self.ema_errors = vec![0.0; n];
        self.ema_initialized = false;
        self.steps_since_update = 0;
        self.total_steps = 0;
        self.current_learning_rate = self.config.learning_rate;
        self.current_regime = ErrorRegime::Stable;
        self.recent.clear();
        self.error_history.clear();
        self.stats = AdaptiveCorrectionStats::default();
        self.stats.current_learning_rate = self.config.learning_rate;
    }

    /// Reset only the momentum (velocity), preserving parameters and stats
    pub fn reset_momentum(&mut self) {
        self.velocity = vec![0.0; self.config.num_parameters];
    }

    /// Reset the learning rate to the initial value
    pub fn reset_learning_rate(&mut self) {
        self.current_learning_rate = self.config.learning_rate;
        self.stats.current_learning_rate = self.config.learning_rate;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_config() -> AdaptiveCorrectionConfig {
        AdaptiveCorrectionConfig {
            num_parameters: 3,
            learning_rate: 0.1,
            learning_rate_decay: 1.0,
            min_learning_rate: 0.001,
            momentum: 0.0,
            cooldown_steps: 1,
            ema_decay: 0.5,
            regime_window: 10,
            trending_threshold: 0.3,
            volatile_threshold: 1.5,
            param_max: 10.0,
            param_min: -10.0,
            window_size: 100,
            warmup_steps: 0,
            trending_multiplier: 1.5,
            mean_reverting_multiplier: 0.5,
            volatile_multiplier: 0.3,
        }
    }

    fn obs(errors: Vec<f64>) -> ErrorObservation {
        ErrorObservation {
            errors,
            gradient_hint: None,
        }
    }

    fn obs_with_hint(errors: Vec<f64>, hint: Vec<f64>) -> ErrorObservation {
        ErrorObservation {
            errors,
            gradient_hint: Some(hint),
        }
    }

    #[test]
    fn test_basic() {
        let instance = AdaptiveCorrection::new();
        assert!(instance.process().is_ok());
    }

    #[test]
    fn test_initial_parameters_zero() {
        let ac = AdaptiveCorrection::with_config(simple_config());
        assert_eq!(ac.parameters().len(), 3);
        for &p in ac.parameters() {
            assert_eq!(p, 0.0);
        }
    }

    #[test]
    fn test_observe_updates_parameters() {
        let mut ac = AdaptiveCorrection::with_config(simple_config());

        let result = ac.observe(&obs(vec![1.0, 0.0, 0.0])).unwrap();
        assert!(result.updated);

        // Parameters should have moved
        assert!(
            ac.parameters()[0].abs() > 0.0,
            "parameter should have been updated, got {}",
            ac.parameters()[0]
        );
    }

    #[test]
    fn test_observe_wrong_dimension_errors() {
        let mut ac = AdaptiveCorrection::with_config(simple_config());
        assert!(ac.observe(&obs(vec![1.0, 2.0])).is_err());
        assert!(ac.observe(&obs(vec![1.0, 2.0, 3.0, 4.0])).is_err());
    }

    #[test]
    fn test_gradient_hint_wrong_dimension() {
        let mut ac = AdaptiveCorrection::with_config(simple_config());
        assert!(
            ac.observe(&obs_with_hint(vec![1.0, 2.0, 3.0], vec![1.0, 2.0]))
                .is_err()
        );
    }

    #[test]
    fn test_gradient_hint_used() {
        let mut ac = AdaptiveCorrection::with_config(AdaptiveCorrectionConfig {
            momentum: 0.0,
            ..simple_config()
        });

        // With explicit gradient hint
        let result = ac
            .observe(&obs_with_hint(vec![0.0, 0.0, 0.0], vec![1.0, 0.0, 0.0]))
            .unwrap();

        // Parameter[0] should have moved in direction of gradient
        assert!(
            result.parameters[0].abs() > 0.0,
            "parameter should move with gradient hint"
        );
    }

    #[test]
    fn test_cooldown_skips_updates() {
        let mut ac = AdaptiveCorrection::with_config(AdaptiveCorrectionConfig {
            cooldown_steps: 5,
            warmup_steps: 0,
            ..simple_config()
        });

        // First observation: should update (steps_since_update starts at 0,
        // but increments to 1 before checking >= cooldown)
        // Actually with cooldown_steps=5, first step has steps_since_update=1 < 5
        let r1 = ac.observe(&obs(vec![1.0, 0.0, 0.0])).unwrap();
        // First observation: steps_since_update becomes 1 which is < 5
        assert!(!r1.updated, "cooldown should prevent first update");

        // Steps 2-4: still in cooldown
        for _ in 0..3 {
            let r = ac.observe(&obs(vec![1.0, 0.0, 0.0])).unwrap();
            assert!(!r.updated);
        }

        // Step 5: cooldown elapsed
        let r5 = ac.observe(&obs(vec![1.0, 0.0, 0.0])).unwrap();
        assert!(r5.updated, "should update after cooldown");
    }

    #[test]
    fn test_warmup_skips_updates() {
        let mut ac = AdaptiveCorrection::with_config(AdaptiveCorrectionConfig {
            warmup_steps: 5,
            cooldown_steps: 1,
            ..simple_config()
        });

        for _ in 0..5 {
            let r = ac.observe(&obs(vec![1.0, 0.0, 0.0])).unwrap();
            assert!(r.in_warmup);
            assert!(!r.updated);
        }

        // After warmup
        let r = ac.observe(&obs(vec![1.0, 0.0, 0.0])).unwrap();
        assert!(!r.in_warmup);
        assert!(r.updated);
    }

    #[test]
    fn test_learning_rate_decay() {
        let mut ac = AdaptiveCorrection::with_config(AdaptiveCorrectionConfig {
            learning_rate: 1.0,
            learning_rate_decay: 0.9,
            min_learning_rate: 0.01,
            ..simple_config()
        });

        ac.observe(&obs(vec![1.0, 0.0, 0.0])).unwrap();
        let lr1 = ac.current_learning_rate();

        ac.observe(&obs(vec![1.0, 0.0, 0.0])).unwrap();
        let lr2 = ac.current_learning_rate();

        assert!(lr2 < lr1, "learning rate should decay: {} -> {}", lr1, lr2);
    }

    #[test]
    fn test_learning_rate_floor() {
        let mut ac = AdaptiveCorrection::with_config(AdaptiveCorrectionConfig {
            learning_rate: 0.001,
            learning_rate_decay: 0.1,
            min_learning_rate: 0.0005,
            ..simple_config()
        });

        for _ in 0..100 {
            ac.observe(&obs(vec![1.0, 0.0, 0.0])).unwrap();
        }

        assert!(
            ac.current_learning_rate() >= 0.0005 - 1e-10,
            "learning rate should not drop below floor: {}",
            ac.current_learning_rate()
        );
    }

    #[test]
    fn test_momentum_accumulation() {
        let mut ac = AdaptiveCorrection::with_config(AdaptiveCorrectionConfig {
            momentum: 0.9,
            learning_rate: 0.1,
            learning_rate_decay: 1.0,
            ..simple_config()
        });

        // Consistent gradient direction should build momentum
        for _ in 0..10 {
            ac.observe(&obs(vec![1.0, 0.0, 0.0])).unwrap();
        }

        assert!(
            ac.velocity_norm() > 0.0,
            "velocity should accumulate with momentum"
        );
    }

    #[test]
    fn test_parameter_bounds_clamped() {
        let mut ac = AdaptiveCorrection::with_config(AdaptiveCorrectionConfig {
            param_max: 0.5,
            param_min: -0.5,
            learning_rate: 10.0,
            learning_rate_decay: 1.0,
            momentum: 0.0,
            ..simple_config()
        });

        // Large error should try to push parameter beyond bounds
        for _ in 0..50 {
            ac.observe(&obs(vec![100.0, 0.0, 0.0])).unwrap();
        }

        for &p in ac.parameters() {
            assert!(
                (-0.5 - 1e-10..=0.5 + 1e-10).contains(&p),
                "parameter {} should be within bounds [-0.5, 0.5]",
                p
            );
        }
        assert!(ac.stats().bound_clamp_count > 0);
    }

    #[test]
    fn test_set_parameter() {
        let mut ac = AdaptiveCorrection::with_config(simple_config());
        ac.set_parameter(0, 5.0).unwrap();
        assert!((ac.parameter(0).unwrap() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_set_parameter_clamped() {
        let mut ac = AdaptiveCorrection::with_config(AdaptiveCorrectionConfig {
            param_max: 1.0,
            param_min: -1.0,
            ..simple_config()
        });

        ac.set_parameter(0, 100.0).unwrap();
        assert!((ac.parameter(0).unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_set_parameter_out_of_range() {
        let mut ac = AdaptiveCorrection::with_config(simple_config());
        assert!(ac.set_parameter(10, 1.0).is_err());
    }

    #[test]
    fn test_set_parameters_all() {
        let mut ac = AdaptiveCorrection::with_config(simple_config());
        ac.set_parameters(&[1.0, 2.0, 3.0]).unwrap();
        assert!((ac.parameter(0).unwrap() - 1.0).abs() < 1e-10);
        assert!((ac.parameter(1).unwrap() - 2.0).abs() < 1e-10);
        assert!((ac.parameter(2).unwrap() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_set_parameters_wrong_count() {
        let mut ac = AdaptiveCorrection::with_config(simple_config());
        assert!(ac.set_parameters(&[1.0, 2.0]).is_err());
    }

    #[test]
    fn test_parameter_out_of_range() {
        let ac = AdaptiveCorrection::with_config(simple_config());
        assert!(ac.parameter(10).is_err());
    }

    #[test]
    fn test_regime_stable_with_small_errors() {
        let mut ac = AdaptiveCorrection::with_config(simple_config());

        for _ in 0..20 {
            ac.observe(&obs(vec![0.01, 0.01, 0.01])).unwrap();
        }

        assert_eq!(ac.current_regime(), ErrorRegime::Stable);
    }

    #[test]
    fn test_regime_trending_detection() {
        let mut ac = AdaptiveCorrection::with_config(AdaptiveCorrectionConfig {
            regime_window: 20,
            trending_threshold: 0.2,
            ..simple_config()
        });

        // Monotonically increasing errors → high positive autocorrelation
        for i in 0..30 {
            ac.observe(&obs(vec![i as f64 * 0.1, 0.0, 0.0])).unwrap();
        }

        assert_eq!(
            ac.current_regime(),
            ErrorRegime::Trending,
            "monotonically increasing errors should be detected as trending"
        );
    }

    #[test]
    fn test_regime_volatile_detection() {
        let mut ac = AdaptiveCorrection::with_config(AdaptiveCorrectionConfig {
            regime_window: 20,
            volatile_threshold: 0.5,
            ..simple_config()
        });

        // Wildly varying errors
        for i in 0..30 {
            let error = if i % 2 == 0 { 0.01 } else { 100.0 };
            ac.observe(&obs(vec![error, 0.0, 0.0])).unwrap();
        }

        assert_eq!(
            ac.current_regime(),
            ErrorRegime::Volatile,
            "wildly varying errors should be detected as volatile"
        );
    }

    #[test]
    fn test_regime_multiplier_applied() {
        let mut ac = AdaptiveCorrection::with_config(AdaptiveCorrectionConfig {
            trending_multiplier: 2.0,
            ..simple_config()
        });

        // Force trending regime
        for i in 0..30 {
            let result = ac.observe(&obs(vec![i as f64 * 0.1, 0.0, 0.0])).unwrap();
            if result.regime == ErrorRegime::Trending {
                assert!(
                    (result.regime_multiplier - 2.0).abs() < 1e-10,
                    "trending regime should apply 2.0 multiplier"
                );
            }
        }
    }

    #[test]
    fn test_stats_tracking() {
        let mut ac = AdaptiveCorrection::with_config(simple_config());

        for _ in 0..10 {
            ac.observe(&obs(vec![1.0, 0.5, 0.0])).unwrap();
        }

        assert_eq!(ac.stats().total_observations, 10);
        assert!(ac.stats().total_updates > 0);
        assert!(ac.stats().sum_abs_error > 0.0);
        assert!(ac.stats().sum_sq_error > 0.0);
    }

    #[test]
    fn test_stats_adaptation_rate() {
        let mut ac = AdaptiveCorrection::with_config(AdaptiveCorrectionConfig {
            cooldown_steps: 2,
            ..simple_config()
        });

        for _ in 0..20 {
            ac.observe(&obs(vec![1.0, 0.0, 0.0])).unwrap();
        }

        let rate = ac.stats().adaptation_rate();
        assert!(
            rate > 0.0 && rate < 1.0,
            "adaptation rate should be between 0 and 1 with cooldown: {}",
            rate
        );
    }

    #[test]
    fn test_stats_mean_abs_error() {
        let mut ac = AdaptiveCorrection::with_config(simple_config());
        ac.observe(&obs(vec![3.0, 4.0, 0.0])).unwrap();

        assert!(
            ac.stats().mean_abs_error() > 0.0,
            "mean abs error should be > 0"
        );
    }

    #[test]
    fn test_stats_defaults() {
        let stats = AdaptiveCorrectionStats::default();
        assert_eq!(stats.mean_abs_error(), 0.0);
        assert_eq!(stats.mse(), 0.0);
        assert_eq!(stats.rmse(), 0.0);
        assert_eq!(stats.mean_delta_magnitude(), 0.0);
        assert_eq!(stats.adaptation_rate(), 0.0);
        assert_eq!(stats.regime_count(ErrorRegime::Stable), 0);
    }

    #[test]
    fn test_stats_dominant_regime() {
        let mut ac = AdaptiveCorrection::with_config(simple_config());

        // Small stable errors
        for _ in 0..20 {
            ac.observe(&obs(vec![0.01, 0.01, 0.01])).unwrap();
        }

        assert_eq!(ac.stats().dominant_regime(), ErrorRegime::Stable);
    }

    #[test]
    fn test_windowed_mean_error() {
        let mut ac = AdaptiveCorrection::with_config(AdaptiveCorrectionConfig {
            window_size: 5,
            ..simple_config()
        });

        for _ in 0..5 {
            ac.observe(&obs(vec![1.0, 0.0, 0.0])).unwrap();
        }

        let wm = ac.windowed_mean_error();
        assert!(
            (wm - 1.0).abs() < 1e-10,
            "windowed mean error should be 1.0, got {}",
            wm
        );
    }

    #[test]
    fn test_windowed_error_std_constant() {
        let mut ac = AdaptiveCorrection::with_config(simple_config());

        for _ in 0..20 {
            ac.observe(&obs(vec![1.0, 0.0, 0.0])).unwrap();
        }

        assert!(
            ac.windowed_error_std() < 1e-10,
            "constant errors should have ~0 std, got {}",
            ac.windowed_error_std()
        );
    }

    #[test]
    fn test_is_improving() {
        let mut ac = AdaptiveCorrection::with_config(AdaptiveCorrectionConfig {
            window_size: 20,
            ..simple_config()
        });

        // First half: large errors
        for _ in 0..10 {
            ac.observe(&obs(vec![10.0, 10.0, 10.0])).unwrap();
        }
        // Second half: small errors
        for _ in 0..10 {
            ac.observe(&obs(vec![1.0, 1.0, 1.0])).unwrap();
        }

        assert!(ac.is_improving());
    }

    #[test]
    fn test_is_worsening() {
        let mut ac = AdaptiveCorrection::with_config(AdaptiveCorrectionConfig {
            window_size: 20,
            ..simple_config()
        });

        // First half: small errors
        for _ in 0..10 {
            ac.observe(&obs(vec![1.0, 1.0, 1.0])).unwrap();
        }
        // Second half: large errors
        for _ in 0..10 {
            ac.observe(&obs(vec![10.0, 10.0, 10.0])).unwrap();
        }

        assert!(ac.is_worsening());
    }

    #[test]
    fn test_not_improving_or_worsening_consistent() {
        let mut ac = AdaptiveCorrection::with_config(AdaptiveCorrectionConfig {
            window_size: 20,
            ..simple_config()
        });

        for _ in 0..20 {
            ac.observe(&obs(vec![5.0, 5.0, 5.0])).unwrap();
        }

        assert!(!ac.is_improving());
        assert!(!ac.is_worsening());
    }

    #[test]
    fn test_not_improving_insufficient_data() {
        let mut ac = AdaptiveCorrection::with_config(simple_config());
        for _ in 0..4 {
            ac.observe(&obs(vec![1.0, 0.0, 0.0])).unwrap();
        }
        assert!(!ac.is_improving());
        assert!(!ac.is_worsening());
    }

    #[test]
    fn test_parameter_norm() {
        let mut ac = AdaptiveCorrection::with_config(simple_config());
        ac.set_parameters(&[3.0, 4.0, 0.0]).unwrap();

        assert!(
            (ac.parameter_norm() - 5.0).abs() < 1e-10,
            "norm of [3,4,0] should be 5, got {}",
            ac.parameter_norm()
        );
    }

    #[test]
    fn test_velocity_norm_zero_initially() {
        let ac = AdaptiveCorrection::with_config(simple_config());
        assert_eq!(ac.velocity_norm(), 0.0);
    }

    #[test]
    fn test_reset() {
        let mut ac = AdaptiveCorrection::with_config(simple_config());

        for _ in 0..20 {
            ac.observe(&obs(vec![1.0, 2.0, 3.0])).unwrap();
        }

        assert!(ac.step_count() > 0);
        assert!(ac.parameter_norm() > 0.0);

        ac.reset();

        assert_eq!(ac.step_count(), 0);
        assert_eq!(ac.update_count(), 0);
        assert_eq!(ac.parameter_norm(), 0.0);
        assert_eq!(ac.velocity_norm(), 0.0);
        assert_eq!(ac.current_regime(), ErrorRegime::Stable);
        assert_eq!(ac.window_count(), 0);
        assert!(!ac.in_warmup()); // warmup_steps=0 in simple_config
    }

    #[test]
    fn test_reset_momentum() {
        let mut ac = AdaptiveCorrection::with_config(AdaptiveCorrectionConfig {
            momentum: 0.9,
            ..simple_config()
        });

        for _ in 0..10 {
            ac.observe(&obs(vec![1.0, 0.0, 0.0])).unwrap();
        }

        assert!(ac.velocity_norm() > 0.0);

        ac.reset_momentum();
        assert_eq!(ac.velocity_norm(), 0.0);

        // Parameters should still be non-zero
        assert!(ac.parameter_norm() > 0.0);
    }

    #[test]
    fn test_reset_learning_rate() {
        let mut ac = AdaptiveCorrection::with_config(AdaptiveCorrectionConfig {
            learning_rate: 1.0,
            learning_rate_decay: 0.5,
            ..simple_config()
        });

        for _ in 0..10 {
            ac.observe(&obs(vec![1.0, 0.0, 0.0])).unwrap();
        }

        assert!(ac.current_learning_rate() < 1.0);

        ac.reset_learning_rate();
        assert!((ac.current_learning_rate() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ema_smoothing() {
        let mut ac = AdaptiveCorrection::with_config(AdaptiveCorrectionConfig {
            ema_decay: 0.5,
            ..simple_config()
        });

        ac.observe(&obs(vec![10.0, 0.0, 0.0])).unwrap();
        let ema1 = ac.smoothed_errors()[0];

        ac.observe(&obs(vec![0.0, 0.0, 0.0])).unwrap();
        let ema2 = ac.smoothed_errors()[0];

        // EMA should be between 0 and 10
        assert!(
            ema2 < ema1,
            "EMA should decrease when error drops: {} -> {}",
            ema1,
            ema2
        );
        assert!(ema2 > 0.0, "EMA should still be > 0 due to smoothing");
    }

    #[test]
    fn test_window_eviction() {
        let mut ac = AdaptiveCorrection::with_config(AdaptiveCorrectionConfig {
            window_size: 5,
            ..simple_config()
        });

        for _ in 0..20 {
            ac.observe(&obs(vec![1.0, 0.0, 0.0])).unwrap();
        }

        assert_eq!(ac.window_count(), 5);
        assert_eq!(ac.step_count(), 20);
    }

    #[test]
    fn test_windowed_mean_empty() {
        let ac = AdaptiveCorrection::with_config(simple_config());
        assert_eq!(ac.windowed_mean_error(), 0.0);
    }

    #[test]
    fn test_windowed_std_empty() {
        let ac = AdaptiveCorrection::with_config(simple_config());
        assert_eq!(ac.windowed_error_std(), 0.0);
    }

    #[test]
    fn test_invalid_config_zero_parameters() {
        let ac = AdaptiveCorrection::with_config(AdaptiveCorrectionConfig {
            num_parameters: 0,
            ..Default::default()
        });
        assert!(ac.process().is_err());
    }

    #[test]
    fn test_invalid_config_zero_learning_rate() {
        let ac = AdaptiveCorrection::with_config(AdaptiveCorrectionConfig {
            learning_rate: 0.0,
            ..Default::default()
        });
        assert!(ac.process().is_err());
    }

    #[test]
    fn test_invalid_config_bad_decay() {
        let ac = AdaptiveCorrection::with_config(AdaptiveCorrectionConfig {
            learning_rate_decay: 0.0,
            ..Default::default()
        });
        assert!(ac.process().is_err());
    }

    #[test]
    fn test_invalid_config_bad_momentum() {
        let ac = AdaptiveCorrection::with_config(AdaptiveCorrectionConfig {
            momentum: 1.0,
            ..Default::default()
        });
        assert!(ac.process().is_err());
    }

    #[test]
    fn test_invalid_config_bad_ema_decay() {
        let ac = AdaptiveCorrection::with_config(AdaptiveCorrectionConfig {
            ema_decay: 0.0,
            ..Default::default()
        });
        assert!(ac.process().is_err());
    }

    #[test]
    fn test_invalid_config_bad_param_bounds() {
        let ac = AdaptiveCorrection::with_config(AdaptiveCorrectionConfig {
            param_max: -1.0,
            param_min: 1.0,
            ..Default::default()
        });
        assert!(ac.process().is_err());
    }

    #[test]
    fn test_invalid_config_zero_window() {
        let ac = AdaptiveCorrection::with_config(AdaptiveCorrectionConfig {
            window_size: 0,
            ..Default::default()
        });
        assert!(ac.process().is_err());
    }

    #[test]
    fn test_invalid_config_zero_regime_window() {
        let ac = AdaptiveCorrection::with_config(AdaptiveCorrectionConfig {
            regime_window: 0,
            ..Default::default()
        });
        assert!(ac.process().is_err());
    }

    #[test]
    fn test_invalid_config_bad_trending_threshold() {
        let ac = AdaptiveCorrection::with_config(AdaptiveCorrectionConfig {
            trending_threshold: 0.0,
            ..Default::default()
        });
        assert!(ac.process().is_err());
    }

    #[test]
    fn test_invalid_config_bad_volatile_threshold() {
        let ac = AdaptiveCorrection::with_config(AdaptiveCorrectionConfig {
            volatile_threshold: 0.0,
            ..Default::default()
        });
        assert!(ac.process().is_err());
    }

    #[test]
    fn test_invalid_config_negative_min_lr() {
        let ac = AdaptiveCorrection::with_config(AdaptiveCorrectionConfig {
            min_learning_rate: -1.0,
            ..Default::default()
        });
        assert!(ac.process().is_err());
    }

    #[test]
    fn test_step_count_and_update_count() {
        let mut ac = AdaptiveCorrection::with_config(AdaptiveCorrectionConfig {
            cooldown_steps: 2,
            ..simple_config()
        });

        for _ in 0..10 {
            ac.observe(&obs(vec![1.0, 0.0, 0.0])).unwrap();
        }

        assert_eq!(ac.step_count(), 10);
        assert!(
            ac.update_count() < ac.step_count(),
            "with cooldown, updates < steps"
        );
        assert!(ac.update_count() > 0, "should have done some updates");
    }

    #[test]
    fn test_steps_since_update_resets() {
        let mut ac = AdaptiveCorrection::with_config(AdaptiveCorrectionConfig {
            cooldown_steps: 1,
            ..simple_config()
        });

        ac.observe(&obs(vec![1.0, 0.0, 0.0])).unwrap();
        assert_eq!(ac.steps_since_update(), 0, "should reset after update");
    }

    #[test]
    fn test_adaptation_converges_toward_zero_error() {
        let mut ac = AdaptiveCorrection::with_config(AdaptiveCorrectionConfig {
            learning_rate: 0.01,
            learning_rate_decay: 1.0,
            momentum: 0.0,
            cooldown_steps: 1,
            warmup_steps: 0,
            ..simple_config()
        });

        // Apply gradient hints that push parameters toward reducing error
        // Gradient = -error, so if error is positive, gradient is negative,
        // which pushes parameters in the negative direction
        for _ in 0..100 {
            ac.observe(&obs(vec![1.0, 0.0, 0.0])).unwrap();
        }

        // Parameters should have moved to compensate for positive errors
        // With gradient = -ema_error (negative of positive errors),
        // parameters should move in the negative direction
        // but the EMA smoothing means they adapt gradually
        assert!(ac.parameter_norm() > 0.0, "parameters should have adapted");
    }

    #[test]
    fn test_in_warmup_flag() {
        let mut ac = AdaptiveCorrection::with_config(AdaptiveCorrectionConfig {
            warmup_steps: 3,
            cooldown_steps: 1,
            ..simple_config()
        });

        assert!(ac.in_warmup());

        for _ in 0..3 {
            let r = ac.observe(&obs(vec![1.0, 0.0, 0.0])).unwrap();
            assert!(r.in_warmup);
        }

        // Step 4: past warmup
        let r = ac.observe(&obs(vec![1.0, 0.0, 0.0])).unwrap();
        assert!(!r.in_warmup);
        assert!(!ac.in_warmup());
    }

    #[test]
    fn test_stats_warmup_skips() {
        let mut ac = AdaptiveCorrection::with_config(AdaptiveCorrectionConfig {
            warmup_steps: 5,
            cooldown_steps: 1,
            ..simple_config()
        });

        for _ in 0..5 {
            ac.observe(&obs(vec![1.0, 0.0, 0.0])).unwrap();
        }

        assert_eq!(ac.stats().warmup_skips, 5);
    }

    #[test]
    fn test_stats_cooldown_skips() {
        let mut ac = AdaptiveCorrection::with_config(AdaptiveCorrectionConfig {
            cooldown_steps: 3,
            warmup_steps: 0,
            ..simple_config()
        });

        for _ in 0..10 {
            ac.observe(&obs(vec![1.0, 0.0, 0.0])).unwrap();
        }

        assert!(
            ac.stats().cooldown_skips > 0,
            "should have some cooldown skips"
        );
    }

    #[test]
    fn test_regime_count_tracking() {
        let mut ac = AdaptiveCorrection::with_config(simple_config());

        for _ in 0..20 {
            ac.observe(&obs(vec![0.01, 0.01, 0.01])).unwrap();
        }

        let total: usize = ac.stats().regime_counts.iter().sum();
        assert_eq!(total, 20, "regime counts should sum to total observations");
    }

    #[test]
    fn test_deltas_zero_when_not_updated() {
        let mut ac = AdaptiveCorrection::with_config(AdaptiveCorrectionConfig {
            cooldown_steps: 100,
            warmup_steps: 0,
            ..simple_config()
        });

        // First step won't update because cooldown hasn't elapsed
        let r = ac.observe(&obs(vec![1.0, 0.0, 0.0])).unwrap();
        assert!(!r.updated);
        for &d in &r.deltas {
            assert_eq!(d, 0.0);
        }
    }

    #[test]
    fn test_max_abs_error_tracked() {
        let mut ac = AdaptiveCorrection::with_config(simple_config());

        ac.observe(&obs(vec![1.0, 0.0, 0.0])).unwrap();
        ac.observe(&obs(vec![5.0, 0.0, 0.0])).unwrap();
        ac.observe(&obs(vec![3.0, 0.0, 0.0])).unwrap();

        assert!(
            (ac.stats().max_abs_error - 5.0).abs() < 1e-10,
            "max abs error should be 5.0, got {}",
            ac.stats().max_abs_error
        );
    }
}
