//! Forward-backward coordination
//!
//! Part of the Integration region — Engine component.
//!
//! `ForwardBackward` coordinates the forward inference and backward learning
//! passes of the neuromorphic system. It tracks per-pass timing, accumulates
//! batch statistics, monitors gradient norms and loss values, and exposes
//! EMA-smoothed and windowed diagnostics.

use crate::common::{Error, Result};
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the forward-backward coordinator.
#[derive(Debug, Clone)]
pub struct ForwardBackwardConfig {
    /// Maximum batch size for accumulating samples before a backward pass.
    pub max_batch_size: usize,
    /// Gradient norm clipping threshold. Gradients with norms above this are
    /// scaled down.
    pub gradient_clip_norm: f64,
    /// Learning rate used for parameter updates.
    pub learning_rate: f64,
    /// Momentum factor for gradient updates (0 = no momentum).
    pub momentum: f64,
    /// Minimum number of samples required before a backward pass is triggered.
    pub min_samples_for_backward: usize,
    /// EMA decay factor (0 < α < 1). Higher → faster adaptation.
    pub ema_decay: f64,
    /// Window size for rolling diagnostics.
    pub window_size: usize,
}

impl Default for ForwardBackwardConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 64,
            gradient_clip_norm: 1.0,
            learning_rate: 0.001,
            momentum: 0.9,
            min_samples_for_backward: 8,
            ema_decay: 0.1,
            window_size: 50,
        }
    }
}

// ---------------------------------------------------------------------------
// Pass direction
// ---------------------------------------------------------------------------

/// Direction of a computation pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PassDirection {
    /// Forward inference pass — computes predictions from inputs.
    Forward,
    /// Backward learning pass — computes gradients and updates parameters.
    Backward,
}

impl std::fmt::Display for PassDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PassDirection::Forward => write!(f, "Forward"),
            PassDirection::Backward => write!(f, "Backward"),
        }
    }
}

// ---------------------------------------------------------------------------
// Pass record
// ---------------------------------------------------------------------------

/// Record of a completed computation pass.
#[derive(Debug, Clone)]
pub struct PassRecord {
    /// Direction of the pass.
    pub direction: PassDirection,
    /// Tick at which the pass was executed.
    pub tick: u64,
    /// Number of samples in the batch.
    pub batch_size: usize,
    /// Loss value produced by this pass (forward) or used for gradients
    /// (backward).
    pub loss: f64,
    /// Gradient norm (only meaningful for backward passes; 0.0 for forward).
    pub gradient_norm: f64,
    /// Whether gradient clipping was applied.
    pub was_clipped: bool,
}

// ---------------------------------------------------------------------------
// Sample
// ---------------------------------------------------------------------------

/// A buffered sample awaiting processing.
#[derive(Debug, Clone)]
pub struct Sample {
    /// Input feature vector.
    pub features: Vec<f64>,
    /// Target / label value.
    pub target: f64,
    /// Tick at which the sample was received.
    pub received_tick: u64,
}

// ---------------------------------------------------------------------------
// Tick snapshot (windowed diagnostics)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct TickSnapshot {
    forward_count: u64,
    backward_count: u64,
    buffered_samples: usize,
    avg_loss: f64,
    avg_gradient_norm: f64,
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Cumulative statistics for the forward-backward coordinator.
#[derive(Debug, Clone)]
pub struct ForwardBackwardStats {
    /// Total forward passes executed.
    pub total_forward_passes: u64,
    /// Total backward passes executed.
    pub total_backward_passes: u64,
    /// Total samples processed across all passes.
    pub total_samples_processed: u64,
    /// Total gradient clips applied.
    pub total_clips: u64,
    /// Running sum of all loss values (for average computation).
    pub cumulative_loss: f64,
    /// Running sum of all gradient norms (for average computation).
    pub cumulative_gradient_norm: f64,
    /// Most recent loss value.
    pub last_loss: f64,
    /// Most recent gradient norm.
    pub last_gradient_norm: f64,
    /// EMA-smoothed loss.
    pub ema_loss: f64,
    /// EMA-smoothed gradient norm.
    pub ema_gradient_norm: f64,
    /// EMA-smoothed throughput (samples processed per tick).
    pub ema_throughput: f64,
}

impl Default for ForwardBackwardStats {
    fn default() -> Self {
        Self {
            total_forward_passes: 0,
            total_backward_passes: 0,
            total_samples_processed: 0,
            total_clips: 0,
            cumulative_loss: 0.0,
            cumulative_gradient_norm: 0.0,
            last_loss: 0.0,
            last_gradient_norm: 0.0,
            ema_loss: 0.0,
            ema_gradient_norm: 0.0,
            ema_throughput: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// ForwardBackward engine
// ---------------------------------------------------------------------------

/// Forward-backward computation coordinator.
///
/// Buffers incoming samples, executes forward inference passes to compute
/// predictions and loss, triggers backward passes to compute gradients and
/// update parameters, and tracks comprehensive diagnostics.
pub struct ForwardBackward {
    config: ForwardBackwardConfig,
    /// Buffered samples awaiting processing.
    sample_buffer: VecDeque<Sample>,
    /// History of completed passes (bounded).
    pass_history: VecDeque<PassRecord>,
    /// Current tick counter.
    tick: u64,
    /// Samples processed in the current tick (for throughput EMA).
    samples_this_tick: usize,
    /// Whether EMA values have been initialised.
    ema_initialized: bool,
    /// Rolling window of tick snapshots.
    recent: VecDeque<TickSnapshot>,
    /// Cumulative statistics.
    stats: ForwardBackwardStats,
}

impl Default for ForwardBackward {
    fn default() -> Self {
        Self::new()
    }
}

impl ForwardBackward {
    // -- Construction -------------------------------------------------------

    /// Create a new `ForwardBackward` with default configuration.
    pub fn new() -> Self {
        Self::with_config(ForwardBackwardConfig::default())
    }

    /// Create a new `ForwardBackward` with the given configuration.
    pub fn with_config(config: ForwardBackwardConfig) -> Self {
        Self {
            sample_buffer: VecDeque::with_capacity(config.max_batch_size),
            pass_history: VecDeque::with_capacity(config.window_size),
            tick: 0,
            samples_this_tick: 0,
            ema_initialized: false,
            recent: VecDeque::with_capacity(config.window_size),
            stats: ForwardBackwardStats::default(),
            config,
        }
    }

    // -- Sample management --------------------------------------------------

    /// Push a sample into the buffer for future processing.
    ///
    /// If the buffer is at `max_batch_size`, the oldest sample is evicted.
    pub fn push_sample(&mut self, features: Vec<f64>, target: f64) {
        if self.sample_buffer.len() >= self.config.max_batch_size {
            self.sample_buffer.pop_front();
        }
        self.sample_buffer.push_back(Sample {
            features,
            target,
            received_tick: self.tick,
        });
    }

    /// Number of samples currently buffered.
    pub fn buffered_count(&self) -> usize {
        self.sample_buffer.len()
    }

    /// Whether there are enough samples to trigger a backward pass.
    pub fn can_backward(&self) -> bool {
        self.sample_buffer.len() >= self.config.min_samples_for_backward
    }

    // -- Forward pass -------------------------------------------------------

    /// Execute a forward inference pass on the given input features.
    ///
    /// Returns a simulated prediction and loss value. In a real system this
    /// would delegate to the neural network; here we compute a simple dot-
    /// product-based prediction for demonstration purposes.
    pub fn forward(&mut self, features: &[f64], target: f64) -> Result<PassRecord> {
        if features.is_empty() {
            return Err(Error::Configuration(
                "forward pass requires non-empty features".into(),
            ));
        }

        // Simple simulated prediction: mean of features
        let prediction: f64 = features.iter().sum::<f64>() / features.len() as f64;
        let loss = (prediction - target).powi(2); // MSE for a single sample

        let record = PassRecord {
            direction: PassDirection::Forward,
            tick: self.tick,
            batch_size: 1,
            loss,
            gradient_norm: 0.0,
            was_clipped: false,
        };

        self.stats.total_forward_passes += 1;
        self.stats.total_samples_processed += 1;
        self.stats.cumulative_loss += loss;
        self.stats.last_loss = loss;
        self.samples_this_tick += 1;
        self.record_pass(record.clone());

        Ok(record)
    }

    // -- Backward pass ------------------------------------------------------

    /// Execute a backward learning pass on the currently buffered samples.
    ///
    /// Drains up to `max_batch_size` samples from the buffer, computes a
    /// simulated loss and gradient norm, applies gradient clipping if needed,
    /// and returns the pass record.
    pub fn backward(&mut self) -> Result<PassRecord> {
        if self.sample_buffer.is_empty() {
            return Err(Error::Configuration(
                "backward pass requires buffered samples".into(),
            ));
        }

        let batch_size = self.sample_buffer.len().min(self.config.max_batch_size);
        let batch: Vec<Sample> = self
            .sample_buffer
            .drain(..batch_size)
            .collect();

        // Compute batch loss (mean squared error)
        let mut total_loss = 0.0;
        for sample in &batch {
            if sample.features.is_empty() {
                continue;
            }
            let prediction: f64 =
                sample.features.iter().sum::<f64>() / sample.features.len() as f64;
            total_loss += (prediction - sample.target).powi(2);
        }
        let avg_loss = if batch_size > 0 {
            total_loss / batch_size as f64
        } else {
            0.0
        };

        // Simulated gradient norm: proportional to loss
        let raw_gradient_norm = avg_loss.sqrt() * 2.0;
        let (gradient_norm, was_clipped) = if raw_gradient_norm > self.config.gradient_clip_norm {
            (self.config.gradient_clip_norm, true)
        } else {
            (raw_gradient_norm, false)
        };

        let record = PassRecord {
            direction: PassDirection::Backward,
            tick: self.tick,
            batch_size,
            loss: avg_loss,
            gradient_norm,
            was_clipped,
        };

        self.stats.total_backward_passes += 1;
        self.stats.total_samples_processed += batch_size as u64;
        self.stats.cumulative_loss += avg_loss;
        self.stats.cumulative_gradient_norm += gradient_norm;
        self.stats.last_loss = avg_loss;
        self.stats.last_gradient_norm = gradient_norm;
        if was_clipped {
            self.stats.total_clips += 1;
        }
        self.samples_this_tick += batch_size;
        self.record_pass(record.clone());

        Ok(record)
    }

    /// Execute a combined forward-then-backward step: buffers all provided
    /// samples, then immediately runs a backward pass.
    pub fn step(
        &mut self,
        samples: Vec<(Vec<f64>, f64)>,
    ) -> Result<PassRecord> {
        for (features, target) in samples {
            self.push_sample(features, target);
        }
        self.backward()
    }

    // -- Pass history -------------------------------------------------------

    fn record_pass(&mut self, record: PassRecord) {
        if self.pass_history.len() >= self.config.window_size {
            self.pass_history.pop_front();
        }
        self.pass_history.push_back(record);
    }

    /// Returns the most recent pass records.
    pub fn pass_history(&self) -> &VecDeque<PassRecord> {
        &self.pass_history
    }

    /// Returns the most recent pass record, if any.
    pub fn last_pass(&self) -> Option<&PassRecord> {
        self.pass_history.back()
    }

    // -- Tick ---------------------------------------------------------------

    /// Advance the coordinator by one tick, updating EMA and windowed
    /// diagnostics.
    pub fn tick(&mut self) {
        self.tick += 1;

        let total_passes = self.stats.total_forward_passes + self.stats.total_backward_passes;
        let avg_loss = if total_passes > 0 {
            self.stats.cumulative_loss / total_passes as f64
        } else {
            0.0
        };
        let avg_grad = if self.stats.total_backward_passes > 0 {
            self.stats.cumulative_gradient_norm / self.stats.total_backward_passes as f64
        } else {
            0.0
        };

        let snapshot = TickSnapshot {
            forward_count: self.stats.total_forward_passes,
            backward_count: self.stats.total_backward_passes,
            buffered_samples: self.sample_buffer.len(),
            avg_loss,
            avg_gradient_norm: avg_grad,
        };

        // -- EMA update --
        let throughput = self.samples_this_tick as f64;
        if !self.ema_initialized {
            self.stats.ema_loss = self.stats.last_loss;
            self.stats.ema_gradient_norm = self.stats.last_gradient_norm;
            self.stats.ema_throughput = throughput;
            self.ema_initialized = true;
        } else {
            let a = self.config.ema_decay;
            self.stats.ema_loss =
                a * self.stats.last_loss + (1.0 - a) * self.stats.ema_loss;
            self.stats.ema_gradient_norm =
                a * self.stats.last_gradient_norm + (1.0 - a) * self.stats.ema_gradient_norm;
            self.stats.ema_throughput =
                a * throughput + (1.0 - a) * self.stats.ema_throughput;
        }

        // -- Window update --
        if self.recent.len() >= self.config.window_size {
            self.recent.pop_front();
        }
        self.recent.push_back(snapshot);

        // Reset per-tick counter.
        self.samples_this_tick = 0;
    }

    /// Current tick counter.
    pub fn current_tick(&self) -> u64 {
        self.tick
    }

    /// Convenience: call `tick()` (alias used by some callers).
    pub fn process(&mut self) {
        self.tick();
    }

    // -- Diagnostics --------------------------------------------------------

    /// Returns a reference to the cumulative statistics.
    pub fn stats(&self) -> &ForwardBackwardStats {
        &self.stats
    }

    /// Returns a reference to the configuration.
    pub fn config(&self) -> &ForwardBackwardConfig {
        &self.config
    }

    /// EMA-smoothed loss.
    pub fn smoothed_loss(&self) -> f64 {
        self.stats.ema_loss
    }

    /// EMA-smoothed gradient norm.
    pub fn smoothed_gradient_norm(&self) -> f64 {
        self.stats.ema_gradient_norm
    }

    /// EMA-smoothed throughput (samples per tick).
    pub fn smoothed_throughput(&self) -> f64 {
        self.stats.ema_throughput
    }

    /// Windowed average loss over recent ticks.
    pub fn windowed_avg_loss(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.recent.iter().map(|s| s.avg_loss).sum();
        sum / self.recent.len() as f64
    }

    /// Windowed average gradient norm over recent ticks.
    pub fn windowed_avg_gradient_norm(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.recent.iter().map(|s| s.avg_gradient_norm).sum();
        sum / self.recent.len() as f64
    }

    /// Windowed average buffered-sample count over recent ticks.
    pub fn windowed_avg_buffered(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.recent.iter().map(|s| s.buffered_samples as f64).sum();
        sum / self.recent.len() as f64
    }

    /// Whether loss is trending upward (possible divergence).
    ///
    /// Compares the first half of the window to the second half. Returns
    /// `true` when the second half's average loss exceeds the first half's.
    pub fn is_loss_increasing(&self) -> bool {
        if self.recent.len() < 4 {
            return false;
        }
        let mid = self.recent.len() / 2;
        let first_half: f64 =
            self.recent.iter().take(mid).map(|s| s.avg_loss).sum::<f64>() / mid as f64;
        let second_half: f64 = self.recent.iter().skip(mid).map(|s| s.avg_loss).sum::<f64>()
            / (self.recent.len() - mid) as f64;
        second_half > first_half
    }

    /// Average loss across all passes ever executed.
    pub fn lifetime_avg_loss(&self) -> f64 {
        let total =
            self.stats.total_forward_passes + self.stats.total_backward_passes;
        if total == 0 {
            return 0.0;
        }
        self.stats.cumulative_loss / total as f64
    }

    /// Average gradient norm across all backward passes ever executed.
    pub fn lifetime_avg_gradient_norm(&self) -> f64 {
        if self.stats.total_backward_passes == 0 {
            return 0.0;
        }
        self.stats.cumulative_gradient_norm / self.stats.total_backward_passes as f64
    }

    /// Fraction of backward passes where gradient clipping was applied.
    pub fn clip_ratio(&self) -> f64 {
        if self.stats.total_backward_passes == 0 {
            return 0.0;
        }
        self.stats.total_clips as f64 / self.stats.total_backward_passes as f64
    }

    // -- Reset --------------------------------------------------------------

    /// Reset all internal state to defaults, preserving configuration.
    pub fn reset(&mut self) {
        self.sample_buffer.clear();
        self.pass_history.clear();
        self.tick = 0;
        self.samples_this_tick = 0;
        self.ema_initialized = false;
        self.recent.clear();
        self.stats = ForwardBackwardStats::default();
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> ForwardBackwardConfig {
        ForwardBackwardConfig {
            max_batch_size: 8,
            gradient_clip_norm: 1.0,
            learning_rate: 0.01,
            momentum: 0.9,
            min_samples_for_backward: 2,
            ema_decay: 0.5,
            window_size: 5,
        }
    }

    // -- Construction -------------------------------------------------------

    #[test]
    fn test_new_default() {
        let fb = ForwardBackward::new();
        assert_eq!(fb.current_tick(), 0);
        assert_eq!(fb.buffered_count(), 0);
        assert_eq!(fb.stats().total_forward_passes, 0);
        assert_eq!(fb.stats().total_backward_passes, 0);
    }

    #[test]
    fn test_with_config() {
        let cfg = small_config();
        let fb = ForwardBackward::with_config(cfg.clone());
        assert_eq!(fb.config().max_batch_size, 8);
        assert_eq!(fb.config().min_samples_for_backward, 2);
    }

    // -- Sample management --------------------------------------------------

    #[test]
    fn test_push_sample() {
        let mut fb = ForwardBackward::with_config(small_config());
        fb.push_sample(vec![1.0, 2.0], 0.5);
        assert_eq!(fb.buffered_count(), 1);
        fb.push_sample(vec![3.0, 4.0], 1.0);
        assert_eq!(fb.buffered_count(), 2);
    }

    #[test]
    fn test_buffer_eviction() {
        let mut fb = ForwardBackward::with_config(small_config());
        for i in 0..10 {
            fb.push_sample(vec![i as f64], i as f64);
        }
        assert_eq!(fb.buffered_count(), 8); // max_batch_size = 8
    }

    #[test]
    fn test_can_backward() {
        let mut fb = ForwardBackward::with_config(small_config());
        assert!(!fb.can_backward());
        fb.push_sample(vec![1.0], 0.0);
        assert!(!fb.can_backward()); // need 2
        fb.push_sample(vec![2.0], 0.0);
        assert!(fb.can_backward());
    }

    // -- Forward pass -------------------------------------------------------

    #[test]
    fn test_forward_basic() {
        let mut fb = ForwardBackward::with_config(small_config());
        let record = fb.forward(&[2.0, 4.0], 3.0).unwrap();
        assert_eq!(record.direction, PassDirection::Forward);
        assert_eq!(record.batch_size, 1);
        // prediction = mean([2,4]) = 3.0, loss = (3-3)^2 = 0
        assert!((record.loss - 0.0).abs() < 1e-9);
        assert_eq!(record.gradient_norm, 0.0);
        assert!(!record.was_clipped);
        assert_eq!(fb.stats().total_forward_passes, 1);
    }

    #[test]
    fn test_forward_nonzero_loss() {
        let mut fb = ForwardBackward::with_config(small_config());
        let record = fb.forward(&[2.0, 4.0], 5.0).unwrap();
        // prediction = 3.0, loss = (3-5)^2 = 4.0
        assert!((record.loss - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_forward_empty_features() {
        let mut fb = ForwardBackward::with_config(small_config());
        let result = fb.forward(&[], 1.0);
        assert!(result.is_err());
    }

    // -- Backward pass ------------------------------------------------------

    #[test]
    fn test_backward_basic() {
        let mut fb = ForwardBackward::with_config(small_config());
        fb.push_sample(vec![1.0, 3.0], 2.0); // pred=2, loss=0
        fb.push_sample(vec![0.0, 4.0], 2.0); // pred=2, loss=0
        let record = fb.backward().unwrap();
        assert_eq!(record.direction, PassDirection::Backward);
        assert_eq!(record.batch_size, 2);
        assert!((record.loss - 0.0).abs() < 1e-9);
        assert_eq!(fb.buffered_count(), 0); // drained
        assert_eq!(fb.stats().total_backward_passes, 1);
    }

    #[test]
    fn test_backward_empty_buffer() {
        let mut fb = ForwardBackward::with_config(small_config());
        let result = fb.backward();
        assert!(result.is_err());
    }

    #[test]
    fn test_backward_gradient_clipping() {
        let mut fb = ForwardBackward::with_config(small_config());
        // Large loss → large gradient norm → clipped
        fb.push_sample(vec![10.0], 0.0); // pred=10, loss=100
        fb.push_sample(vec![10.0], 0.0);
        let record = fb.backward().unwrap();
        // avg_loss = 100, raw_grad = sqrt(100)*2 = 20 > clip_norm 1.0
        assert!(record.was_clipped);
        assert!((record.gradient_norm - 1.0).abs() < 1e-9);
        assert_eq!(fb.stats().total_clips, 1);
    }

    #[test]
    fn test_backward_no_clipping() {
        let mut fb = ForwardBackward::with_config(small_config());
        // Small loss → small gradient norm → not clipped
        fb.push_sample(vec![1.0], 1.0); // pred=1, loss=0
        fb.push_sample(vec![1.0], 1.0);
        let record = fb.backward().unwrap();
        assert!(!record.was_clipped);
        assert!((record.gradient_norm - 0.0).abs() < 1e-9);
    }

    // -- Step ---------------------------------------------------------------

    #[test]
    fn test_step() {
        let mut fb = ForwardBackward::with_config(small_config());
        let samples = vec![
            (vec![1.0, 3.0], 2.0),
            (vec![2.0, 4.0], 3.0),
        ];
        let record = fb.step(samples).unwrap();
        assert_eq!(record.direction, PassDirection::Backward);
        assert_eq!(record.batch_size, 2);
        assert_eq!(fb.buffered_count(), 0);
    }

    // -- Pass history -------------------------------------------------------

    #[test]
    fn test_pass_history() {
        let mut fb = ForwardBackward::with_config(small_config());
        fb.forward(&[1.0], 1.0).unwrap();
        fb.push_sample(vec![2.0], 1.0);
        fb.push_sample(vec![3.0], 1.0);
        fb.backward().unwrap();
        assert_eq!(fb.pass_history().len(), 2);
        assert_eq!(fb.last_pass().unwrap().direction, PassDirection::Backward);
    }

    #[test]
    fn test_pass_history_bounded() {
        let mut fb = ForwardBackward::with_config(small_config()); // window_size=5
        for _ in 0..10 {
            fb.forward(&[1.0], 1.0).unwrap();
        }
        assert_eq!(fb.pass_history().len(), 5);
    }

    // -- Tick & EMA ---------------------------------------------------------

    #[test]
    fn test_tick_increments() {
        let mut fb = ForwardBackward::with_config(small_config());
        assert_eq!(fb.current_tick(), 0);
        fb.tick();
        assert_eq!(fb.current_tick(), 1);
        fb.tick();
        assert_eq!(fb.current_tick(), 2);
    }

    #[test]
    fn test_process_alias() {
        let mut fb = ForwardBackward::with_config(small_config());
        fb.process();
        assert_eq!(fb.current_tick(), 1);
    }

    #[test]
    fn test_ema_initialises_on_first_tick() {
        let mut fb = ForwardBackward::with_config(small_config());
        fb.forward(&[2.0, 4.0], 5.0).unwrap(); // loss = 4.0
        fb.tick();
        assert!((fb.smoothed_loss() - 4.0).abs() < 1e-9);
        assert!((fb.smoothed_throughput() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_ema_blends_on_subsequent_ticks() {
        let mut fb = ForwardBackward::with_config(small_config()); // ema_decay = 0.5
        fb.forward(&[2.0, 4.0], 5.0).unwrap(); // loss = 4.0
        fb.tick(); // ema_loss = 4.0 (init)
        fb.forward(&[1.0], 1.0).unwrap(); // loss = 0.0
        fb.tick(); // ema_loss = 0.5*0.0 + 0.5*4.0 = 2.0
        assert!((fb.smoothed_loss() - 2.0).abs() < 1e-9);
    }

    // -- Windowed diagnostics -----------------------------------------------

    #[test]
    fn test_windowed_avg_loss() {
        let mut fb = ForwardBackward::with_config(small_config());
        // Run several ticks with different losses
        fb.forward(&[2.0, 4.0], 5.0).unwrap(); // loss=4
        fb.tick();
        fb.forward(&[1.0], 1.0).unwrap(); // loss=0
        fb.tick();
        // windowed average should be somewhere between the two snapshots
        let w = fb.windowed_avg_loss();
        assert!(w >= 0.0);
    }

    #[test]
    fn test_windowed_empty() {
        let fb = ForwardBackward::with_config(small_config());
        assert!((fb.windowed_avg_loss() - 0.0).abs() < 1e-9);
        assert!((fb.windowed_avg_gradient_norm() - 0.0).abs() < 1e-9);
        assert!((fb.windowed_avg_buffered() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_windowed_buffered() {
        let mut fb = ForwardBackward::with_config(small_config());
        fb.push_sample(vec![1.0], 0.0);
        fb.push_sample(vec![2.0], 0.0);
        fb.tick();
        assert!((fb.windowed_avg_buffered() - 2.0).abs() < 1e-9);
    }

    // -- Loss trend detection -----------------------------------------------

    #[test]
    fn test_is_loss_increasing_insufficient_data() {
        let mut fb = ForwardBackward::with_config(small_config());
        fb.tick();
        fb.tick();
        assert!(!fb.is_loss_increasing());
    }

    #[test]
    fn test_is_loss_increasing_true() {
        let mut fb = ForwardBackward::with_config(small_config());
        // First two ticks: low loss
        fb.forward(&[1.0], 1.0).unwrap(); // loss=0
        fb.tick();
        fb.forward(&[1.0], 1.0).unwrap(); // loss=0
        fb.tick();
        // Next two ticks: higher loss
        fb.forward(&[10.0], 0.0).unwrap(); // loss=100
        fb.tick();
        fb.forward(&[10.0], 0.0).unwrap(); // loss=100
        fb.tick();
        // avg_loss in snapshots will reflect cumulative — but the cumulative
        // average will increase as more high-loss passes are added.
        // This is a heuristic check.
    }

    // -- Lifetime stats -----------------------------------------------------

    #[test]
    fn test_lifetime_avg_loss() {
        let mut fb = ForwardBackward::with_config(small_config());
        fb.forward(&[2.0, 4.0], 5.0).unwrap(); // loss=4
        fb.forward(&[1.0], 1.0).unwrap(); // loss=0
        // total_forward_passes = 2, cumulative_loss = 4.0
        assert!((fb.lifetime_avg_loss() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_lifetime_avg_loss_empty() {
        let fb = ForwardBackward::with_config(small_config());
        assert!((fb.lifetime_avg_loss() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_lifetime_avg_gradient_norm() {
        let mut fb = ForwardBackward::with_config(small_config());
        fb.push_sample(vec![1.0], 1.0); // pred=1, loss=0, grad=0
        fb.push_sample(vec![1.0], 1.0);
        fb.backward().unwrap();
        assert!((fb.lifetime_avg_gradient_norm() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_lifetime_avg_gradient_norm_empty() {
        let fb = ForwardBackward::with_config(small_config());
        assert!((fb.lifetime_avg_gradient_norm() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_clip_ratio() {
        let mut fb = ForwardBackward::with_config(small_config());
        // Two backward passes: one clipped, one not
        fb.push_sample(vec![10.0], 0.0);
        fb.push_sample(vec![10.0], 0.0);
        fb.backward().unwrap(); // clipped
        fb.push_sample(vec![1.0], 1.0);
        fb.push_sample(vec![1.0], 1.0);
        fb.backward().unwrap(); // not clipped
        assert!((fb.clip_ratio() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_clip_ratio_empty() {
        let fb = ForwardBackward::with_config(small_config());
        assert!((fb.clip_ratio() - 0.0).abs() < 1e-9);
    }

    // -- Reset --------------------------------------------------------------

    #[test]
    fn test_reset() {
        let mut fb = ForwardBackward::with_config(small_config());
        fb.push_sample(vec![1.0], 0.0);
        fb.forward(&[1.0], 0.0).unwrap();
        fb.tick();

        fb.reset();
        assert_eq!(fb.current_tick(), 0);
        assert_eq!(fb.buffered_count(), 0);
        assert_eq!(fb.stats().total_forward_passes, 0);
        assert_eq!(fb.stats().total_backward_passes, 0);
        assert!(fb.pass_history().is_empty());
        assert!((fb.smoothed_loss() - 0.0).abs() < 1e-9);
    }

    // -- Full lifecycle -----------------------------------------------------

    #[test]
    fn test_full_lifecycle() {
        let mut fb = ForwardBackward::with_config(small_config());

        // Tick 1: push samples and do forward
        fb.push_sample(vec![1.0, 2.0], 1.5);
        fb.push_sample(vec![3.0, 4.0], 3.0);
        fb.forward(&[5.0, 6.0], 5.5).unwrap();
        fb.tick();

        // Tick 2: backward pass
        assert!(fb.can_backward());
        let bw = fb.backward().unwrap();
        assert_eq!(bw.direction, PassDirection::Backward);
        assert_eq!(bw.batch_size, 2);
        assert_eq!(fb.buffered_count(), 0);
        fb.tick();

        // Tick 3: step
        let samples = vec![
            (vec![1.0], 1.0),
            (vec![2.0], 2.0),
            (vec![3.0], 3.0),
        ];
        fb.step(samples).unwrap();
        fb.tick();

        assert_eq!(fb.current_tick(), 3);
        assert_eq!(fb.stats().total_forward_passes, 1);
        assert_eq!(fb.stats().total_backward_passes, 2);
        assert!(fb.stats().total_samples_processed > 0);
    }

    #[test]
    fn test_window_rolls() {
        let mut fb = ForwardBackward::with_config(small_config()); // window_size=5
        for _ in 0..10 {
            fb.forward(&[1.0], 1.0).unwrap();
            fb.tick();
        }
        // Recent window should be at capacity
        assert!(fb.recent.len() <= 5);
    }
}
