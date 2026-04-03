//! Attention gating mechanism
//!
//! Part of the Thalamus region
//! Component: attention
//!
//! Implements an attention gate that selectively passes or blocks signals
//! based on a learned activation threshold. This models the thalamic
//! reticular nucleus's ability to gate information flow between brain
//! regions.
//!
//! ## Features
//!
//! - **Threshold-based gating**: Signals above the activation threshold
//!   pass through (scaled by gain); signals below are suppressed.
//! - **Adaptive gain**: The gate's gain adjusts based on recent signal
//!   statistics, amplifying weak but relevant signals and attenuating
//!   noise.
//! - **Hysteresis**: A configurable hysteresis band prevents rapid
//!   toggling when signals hover near the threshold.
//! - **Soft gating mode**: Instead of hard on/off, applies a sigmoid
//!   transfer function for smooth gating.
//! - **Multi-channel support**: Gates multiple independent signal
//!   channels simultaneously with per-channel state.
//! - **EMA-smoothed gate state**: Tracks smoothed open-fraction for
//!   downstream monitoring.
//! - **Per-channel statistics**: Tracks pass/block rates, mean signal
//!   levels, and gate duty cycle.

use std::collections::VecDeque;

use crate::common::{Error, Result};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the attention gate.
#[derive(Debug, Clone)]
pub struct GateConfig {
    /// Number of independent signal channels.
    pub num_channels: usize,
    /// Base activation threshold. Signals above this pass; below are blocked.
    pub threshold: f64,
    /// Hysteresis band width (applied symmetrically around the threshold).
    /// When the gate is open, it stays open until the signal drops below
    /// `threshold - hysteresis`. When closed, it stays closed until the
    /// signal rises above `threshold + hysteresis`.
    pub hysteresis: f64,
    /// Gain multiplier applied to signals that pass through the gate.
    pub gain: f64,
    /// Whether to use soft (sigmoid) gating instead of hard thresholding.
    pub soft_gating: bool,
    /// Steepness of the sigmoid transfer function when `soft_gating` is true.
    /// Higher values → sharper transition (more like hard gating).
    pub sigmoid_steepness: f64,
    /// Whether to adaptively adjust the gain based on recent signal statistics.
    pub adaptive_gain: bool,
    /// EMA decay for the adaptive gain estimator (0, 1).
    pub gain_ema_decay: f64,
    /// Target mean output level for the adaptive gain controller.
    pub target_output_level: f64,
    /// EMA decay for the smoothed open-fraction tracker (0, 1).
    /// Set to 0 to disable.
    pub ema_decay: f64,
    /// Maximum number of recent gate-state snapshots to retain for windowed
    /// statistics.
    pub window_size: usize,
    /// Maximum gain to prevent runaway amplification.
    pub max_gain: f64,
    /// Minimum gain to prevent complete signal suppression.
    pub min_gain: f64,
}

impl Default for GateConfig {
    fn default() -> Self {
        Self {
            num_channels: 1,
            threshold: 0.5,
            hysteresis: 0.05,
            gain: 1.0,
            soft_gating: false,
            sigmoid_steepness: 10.0,
            adaptive_gain: false,
            gain_ema_decay: 0.9,
            target_output_level: 0.5,
            ema_decay: 0.8,
            window_size: 200,
            max_gain: 10.0,
            min_gain: 0.01,
        }
    }
}

// ---------------------------------------------------------------------------
// Input / Output types
// ---------------------------------------------------------------------------

/// Input signals to the gate (one value per channel).
#[derive(Debug, Clone)]
pub struct GateInput {
    /// Signal values, one per channel.
    pub signals: Vec<f64>,
}

/// Output of the gate.
#[derive(Debug, Clone)]
pub struct GateOutput {
    /// Gated signal values, one per channel.
    pub signals: Vec<f64>,
    /// Per-channel gate state: 1.0 = fully open, 0.0 = fully closed.
    /// For soft gating, this is the sigmoid output.
    pub gate_values: Vec<f64>,
    /// Fraction of channels that are open (gate_value > 0.5).
    pub open_fraction: f64,
    /// EMA-smoothed open fraction.
    pub smoothed_open_fraction: f64,
    /// Current effective gain (may differ from configured gain if adaptive).
    pub effective_gain: f64,
}

/// Per-channel statistics.
#[derive(Debug, Clone)]
pub struct ChannelStats {
    /// Total signals processed on this channel.
    pub total_signals: usize,
    /// Number of signals that passed (gate open).
    pub pass_count: usize,
    /// Number of signals that were blocked (gate closed).
    pub block_count: usize,
    /// EMA-smoothed signal level.
    pub ema_signal: f64,
    /// EMA-smoothed output level.
    pub ema_output: f64,
    /// Gate duty cycle (fraction of time open).
    pub duty_cycle: f64,
}

/// Aggregate statistics for the gate.
#[derive(Debug, Clone, Default)]
pub struct GateStats {
    /// Total gate operations (forward passes).
    pub total_operations: usize,
    /// Cumulative sum of open fractions (for averaging).
    pub sum_open_fraction: f64,
    /// Number of gate state transitions (open→closed or closed→open)
    /// across all channels.
    pub total_transitions: usize,
}

impl GateStats {
    /// Average open fraction across all operations.
    pub fn avg_open_fraction(&self) -> f64 {
        if self.total_operations == 0 {
            0.0
        } else {
            self.sum_open_fraction / self.total_operations as f64
        }
    }
}

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct ChannelState {
    is_open: bool,
    ema_signal: f64,
    ema_output: f64,
    signal_initialized: bool,
    total_signals: usize,
    pass_count: usize,
    block_count: usize,
}

impl ChannelState {
    fn new() -> Self {
        Self {
            is_open: false,
            ema_signal: 0.0,
            ema_output: 0.0,
            signal_initialized: false,
            total_signals: 0,
            pass_count: 0,
            block_count: 0,
        }
    }

    fn duty_cycle(&self) -> f64 {
        if self.total_signals == 0 {
            0.0
        } else {
            self.pass_count as f64 / self.total_signals as f64
        }
    }
}

// ---------------------------------------------------------------------------
// Gate
// ---------------------------------------------------------------------------

/// Attention gating mechanism.
///
/// Call [`forward`] with a set of signal values to gate them based on the
/// configured threshold, hysteresis, and gain settings.
pub struct Gate {
    config: GateConfig,
    /// Per-channel state.
    channels: Vec<ChannelState>,
    /// Current effective gain.
    effective_gain: f64,
    /// EMA state for smoothed open fraction.
    ema_open_fraction: f64,
    ema_initialized: bool,
    /// Windowed history of open fractions.
    history: VecDeque<f64>,
    /// Running statistics.
    stats: GateStats,
}

impl Default for Gate {
    fn default() -> Self {
        Self::new()
    }
}

impl Gate {
    /// Create a new instance with default configuration.
    pub fn new() -> Self {
        Self::with_config(GateConfig::default())
    }

    /// Create a new instance with the given configuration.
    pub fn with_config(config: GateConfig) -> Self {
        let channels = vec![ChannelState::new(); config.num_channels];
        let gain = config.gain;
        Self {
            channels,
            effective_gain: gain,
            ema_open_fraction: 0.0,
            ema_initialized: false,
            history: VecDeque::with_capacity(config.window_size),
            stats: GateStats::default(),
            config,
        }
    }

    /// Validate configuration parameters.
    pub fn process(&self) -> Result<()> {
        if self.config.num_channels == 0 {
            return Err(Error::InvalidInput("num_channels must be > 0".into()));
        }
        if self.config.hysteresis < 0.0 {
            return Err(Error::InvalidInput("hysteresis must be >= 0".into()));
        }
        if self.config.gain <= 0.0 {
            return Err(Error::InvalidInput("gain must be > 0".into()));
        }
        if self.config.sigmoid_steepness <= 0.0 {
            return Err(Error::InvalidInput("sigmoid_steepness must be > 0".into()));
        }
        if self.config.gain_ema_decay <= 0.0 || self.config.gain_ema_decay >= 1.0 {
            return Err(Error::InvalidInput(
                "gain_ema_decay must be in (0, 1)".into(),
            ));
        }
        if self.config.target_output_level <= 0.0 {
            return Err(Error::InvalidInput(
                "target_output_level must be > 0".into(),
            ));
        }
        if self.config.ema_decay < 0.0 || self.config.ema_decay >= 1.0 {
            return Err(Error::InvalidInput("ema_decay must be in [0, 1)".into()));
        }
        if self.config.window_size == 0 {
            return Err(Error::InvalidInput("window_size must be > 0".into()));
        }
        if self.config.max_gain <= 0.0 {
            return Err(Error::InvalidInput("max_gain must be > 0".into()));
        }
        if self.config.min_gain < 0.0 {
            return Err(Error::InvalidInput("min_gain must be >= 0".into()));
        }
        if self.config.max_gain < self.config.min_gain {
            return Err(Error::InvalidInput("max_gain must be >= min_gain".into()));
        }
        Ok(())
    }

    // -- Forward pass ------------------------------------------------------

    /// Gate the input signals.
    pub fn forward(&mut self, input: &GateInput) -> Result<GateOutput> {
        if input.signals.len() != self.config.num_channels {
            return Err(Error::InvalidInput(format!(
                "expected {} channels, got {}",
                self.config.num_channels,
                input.signals.len()
            )));
        }

        let mut gated_signals = vec![0.0_f64; self.config.num_channels];
        let mut gate_values = vec![0.0_f64; self.config.num_channels];
        let mut open_count = 0usize;

        for (ch, &signal) in input.signals.iter().enumerate() {
            let state = &mut self.channels[ch];
            state.total_signals += 1;

            // Update EMA of signal level.
            let alpha = self.config.gain_ema_decay;
            if state.signal_initialized {
                state.ema_signal = alpha * state.ema_signal + (1.0 - alpha) * signal.abs();
            } else {
                state.ema_signal = signal.abs();
                state.signal_initialized = true;
            }

            // Determine gate value.
            let gate_val = if self.config.soft_gating {
                // Sigmoid: 1 / (1 + exp(-steepness * (signal - threshold)))
                let x = self.config.sigmoid_steepness * (signal - self.config.threshold);
                1.0 / (1.0 + (-x).exp())
            } else {
                // Hard gating with hysteresis.
                let was_open = state.is_open;
                let new_open = if was_open {
                    signal >= self.config.threshold - self.config.hysteresis
                } else {
                    signal > self.config.threshold + self.config.hysteresis
                };

                if was_open != new_open {
                    self.stats.total_transitions += 1;
                }
                state.is_open = new_open;

                if new_open { 1.0 } else { 0.0 }
            };

            gate_values[ch] = gate_val;

            let output = signal * gate_val * self.effective_gain;
            gated_signals[ch] = output;

            // Update EMA of output level.
            if state.signal_initialized {
                state.ema_output = alpha * state.ema_output + (1.0 - alpha) * output.abs();
            }

            if gate_val > 0.5 {
                open_count += 1;
                state.pass_count += 1;
            } else {
                state.block_count += 1;
            }
        }

        let open_fraction = open_count as f64 / self.config.num_channels as f64;

        // Adaptive gain: adjust gain to match target output level.
        if self.config.adaptive_gain {
            let mean_output: f64 = self
                .channels
                .iter()
                .filter(|c| c.signal_initialized)
                .map(|c| c.ema_output)
                .sum::<f64>()
                / self.config.num_channels.max(1) as f64;

            if mean_output > 1e-12 {
                let ratio = self.config.target_output_level / mean_output;
                // Smooth adjustment towards target.
                let alpha = self.config.gain_ema_decay;
                self.effective_gain =
                    alpha * self.effective_gain + (1.0 - alpha) * (self.effective_gain * ratio);
            }
            self.effective_gain = self
                .effective_gain
                .max(self.config.min_gain)
                .min(self.config.max_gain);
        }

        // EMA smoothing of open fraction.
        let smoothed = if self.config.ema_decay > 0.0 {
            if self.ema_initialized {
                let s = self.config.ema_decay * self.ema_open_fraction
                    + (1.0 - self.config.ema_decay) * open_fraction;
                self.ema_open_fraction = s;
                s
            } else {
                self.ema_open_fraction = open_fraction;
                self.ema_initialized = true;
                open_fraction
            }
        } else {
            open_fraction
        };

        // History.
        self.history.push_back(open_fraction);
        while self.history.len() > self.config.window_size {
            self.history.pop_front();
        }

        self.stats.total_operations += 1;
        self.stats.sum_open_fraction += open_fraction;

        Ok(GateOutput {
            signals: gated_signals,
            gate_values,
            open_fraction,
            smoothed_open_fraction: smoothed,
            effective_gain: self.effective_gain,
        })
    }

    // -- Accessors ---------------------------------------------------------

    /// Get per-channel statistics.
    pub fn channel_stats(&self) -> Vec<ChannelStats> {
        self.channels
            .iter()
            .map(|c| ChannelStats {
                total_signals: c.total_signals,
                pass_count: c.pass_count,
                block_count: c.block_count,
                ema_signal: c.ema_signal,
                ema_output: c.ema_output,
                duty_cycle: c.duty_cycle(),
            })
            .collect()
    }

    /// Get aggregate statistics.
    pub fn stats(&self) -> &GateStats {
        &self.stats
    }

    /// Current effective gain.
    pub fn effective_gain(&self) -> f64 {
        self.effective_gain
    }

    /// Number of channels.
    pub fn num_channels(&self) -> usize {
        self.config.num_channels
    }

    /// Whether a specific channel is currently open (hard gating only).
    pub fn is_open(&self, channel: usize) -> bool {
        self.channels
            .get(channel)
            .map(|c| c.is_open)
            .unwrap_or(false)
    }

    /// Windowed mean of recent open fractions.
    pub fn windowed_mean(&self) -> Option<f64> {
        if self.history.is_empty() {
            return None;
        }
        let sum: f64 = self.history.iter().sum();
        Some(sum / self.history.len() as f64)
    }

    /// Windowed standard deviation of recent open fractions.
    pub fn windowed_std(&self) -> Option<f64> {
        if self.history.len() < 2 {
            return None;
        }
        let mean = self.windowed_mean().unwrap();
        let var: f64 = self.history.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
            / (self.history.len() - 1) as f64;
        Some(var.sqrt())
    }

    /// Set the threshold dynamically.
    pub fn set_threshold(&mut self, threshold: f64) {
        self.config.threshold = threshold;
    }

    /// Set the gain dynamically.
    pub fn set_gain(&mut self, gain: f64) {
        self.effective_gain = gain.clamp(self.config.min_gain, self.config.max_gain);
        self.config.gain = self.effective_gain;
    }

    /// Reset all state (channels, EMA, history, stats).
    pub fn reset(&mut self) {
        self.channels = vec![ChannelState::new(); self.config.num_channels];
        self.effective_gain = self.config.gain;
        self.ema_open_fraction = 0.0;
        self.ema_initialized = false;
        self.history.clear();
        self.stats = GateStats::default();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn input(signals: Vec<f64>) -> GateInput {
        GateInput { signals }
    }

    fn default_config() -> GateConfig {
        GateConfig {
            num_channels: 3,
            threshold: 0.5,
            hysteresis: 0.05,
            gain: 1.0,
            soft_gating: false,
            sigmoid_steepness: 10.0,
            adaptive_gain: false,
            gain_ema_decay: 0.9,
            target_output_level: 0.5,
            ema_decay: 0.5,
            window_size: 100,
            max_gain: 10.0,
            min_gain: 0.01,
        }
    }

    fn default_gate() -> Gate {
        Gate::with_config(default_config())
    }

    // -- Config validation -------------------------------------------------

    #[test]
    fn test_basic() {
        let instance = Gate::new();
        assert!(instance.process().is_ok());
    }

    #[test]
    fn test_process_invalid_num_channels() {
        let g = Gate::with_config(GateConfig {
            num_channels: 0,
            ..Default::default()
        });
        assert!(g.process().is_err());
    }

    #[test]
    fn test_process_invalid_hysteresis() {
        let g = Gate::with_config(GateConfig {
            hysteresis: -1.0,
            ..Default::default()
        });
        assert!(g.process().is_err());
    }

    #[test]
    fn test_process_invalid_gain() {
        let g = Gate::with_config(GateConfig {
            gain: 0.0,
            ..Default::default()
        });
        assert!(g.process().is_err());
    }

    #[test]
    fn test_process_invalid_sigmoid_steepness() {
        let g = Gate::with_config(GateConfig {
            sigmoid_steepness: 0.0,
            ..Default::default()
        });
        assert!(g.process().is_err());
    }

    #[test]
    fn test_process_invalid_gain_ema_decay() {
        let g = Gate::with_config(GateConfig {
            gain_ema_decay: 0.0,
            ..Default::default()
        });
        assert!(g.process().is_err());
    }

    #[test]
    fn test_process_invalid_target_output_level() {
        let g = Gate::with_config(GateConfig {
            target_output_level: 0.0,
            ..Default::default()
        });
        assert!(g.process().is_err());
    }

    #[test]
    fn test_process_invalid_ema_decay() {
        let g = Gate::with_config(GateConfig {
            ema_decay: 1.0,
            ..Default::default()
        });
        assert!(g.process().is_err());
    }

    #[test]
    fn test_process_invalid_window_size() {
        let g = Gate::with_config(GateConfig {
            window_size: 0,
            ..Default::default()
        });
        assert!(g.process().is_err());
    }

    #[test]
    fn test_process_invalid_max_gain() {
        let g = Gate::with_config(GateConfig {
            max_gain: 0.0,
            ..Default::default()
        });
        assert!(g.process().is_err());
    }

    #[test]
    fn test_process_invalid_min_gain() {
        let g = Gate::with_config(GateConfig {
            min_gain: -1.0,
            ..Default::default()
        });
        assert!(g.process().is_err());
    }

    #[test]
    fn test_process_invalid_max_less_than_min() {
        let g = Gate::with_config(GateConfig {
            max_gain: 1.0,
            min_gain: 5.0,
            ..Default::default()
        });
        assert!(g.process().is_err());
    }

    #[test]
    fn test_process_valid_ema_decay_zero() {
        let g = Gate::with_config(GateConfig {
            ema_decay: 0.0,
            ..Default::default()
        });
        assert!(g.process().is_ok());
    }

    // -- Input validation --------------------------------------------------

    #[test]
    fn test_forward_wrong_channel_count() {
        let mut g = default_gate();
        assert!(g.forward(&input(vec![1.0, 2.0])).is_err()); // expects 3
    }

    // -- Hard gating -------------------------------------------------------

    #[test]
    fn test_hard_gate_above_threshold() {
        let mut g = Gate::with_config(GateConfig {
            num_channels: 1,
            threshold: 0.5,
            hysteresis: 0.0,
            gain: 1.0,
            ..default_config()
        });
        let out = g.forward(&input(vec![0.8])).unwrap();
        assert!(
            (out.signals[0] - 0.8).abs() < 1e-10,
            "signal above threshold should pass, got {}",
            out.signals[0]
        );
        assert!((out.gate_values[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_hard_gate_below_threshold() {
        let mut g = Gate::with_config(GateConfig {
            num_channels: 1,
            threshold: 0.5,
            hysteresis: 0.0,
            gain: 1.0,
            ..default_config()
        });
        let out = g.forward(&input(vec![0.3])).unwrap();
        assert!(
            out.signals[0].abs() < 1e-10,
            "signal below threshold should be blocked, got {}",
            out.signals[0]
        );
        assert!(out.gate_values[0].abs() < 1e-10);
    }

    #[test]
    fn test_hard_gate_at_threshold() {
        let mut g = Gate::with_config(GateConfig {
            num_channels: 1,
            threshold: 0.5,
            hysteresis: 0.0,
            gain: 1.0,
            ..default_config()
        });
        // Exactly at threshold with no hysteresis: should stay closed
        // (need strictly above threshold + hysteresis to open)
        let out = g.forward(&input(vec![0.5])).unwrap();
        assert!(
            out.gate_values[0] < 0.5,
            "at threshold with no hysteresis, gate should remain closed"
        );
    }

    #[test]
    fn test_hard_gate_gain_applied() {
        let mut g = Gate::with_config(GateConfig {
            num_channels: 1,
            threshold: 0.5,
            hysteresis: 0.0,
            gain: 2.0,
            ..default_config()
        });
        let out = g.forward(&input(vec![0.8])).unwrap();
        assert!(
            (out.signals[0] - 1.6).abs() < 1e-10,
            "gain should be applied: expected 1.6, got {}",
            out.signals[0]
        );
    }

    #[test]
    fn test_hard_gate_multi_channel() {
        let mut g = Gate::with_config(GateConfig {
            num_channels: 3,
            threshold: 0.5,
            hysteresis: 0.0,
            gain: 1.0,
            ..default_config()
        });
        let out = g.forward(&input(vec![0.8, 0.2, 0.6])).unwrap();
        assert!(out.gate_values[0] > 0.5); // open
        assert!(out.gate_values[1] < 0.5); // closed
        assert!(out.gate_values[2] > 0.5); // open
        assert!(
            (out.open_fraction - 2.0 / 3.0).abs() < 1e-10,
            "2 of 3 channels should be open"
        );
    }

    // -- Hysteresis --------------------------------------------------------

    #[test]
    fn test_hysteresis_prevents_toggling() {
        let mut g = Gate::with_config(GateConfig {
            num_channels: 1,
            threshold: 0.5,
            hysteresis: 0.1,
            gain: 1.0,
            ..default_config()
        });

        // Signal above threshold+hysteresis → opens gate
        g.forward(&input(vec![0.7])).unwrap();
        assert!(g.is_open(0), "gate should be open after signal 0.7");

        // Signal drops to threshold-0.05 (within hysteresis band) → stays open
        g.forward(&input(vec![0.45])).unwrap();
        assert!(g.is_open(0), "gate should stay open within hysteresis band");

        // Signal drops below threshold-hysteresis → closes gate
        g.forward(&input(vec![0.35])).unwrap();
        assert!(
            !g.is_open(0),
            "gate should close below threshold-hysteresis"
        );
    }

    #[test]
    fn test_hysteresis_close_to_open() {
        let mut g = Gate::with_config(GateConfig {
            num_channels: 1,
            threshold: 0.5,
            hysteresis: 0.1,
            gain: 1.0,
            ..default_config()
        });

        // Signal within hysteresis band while closed → stays closed
        g.forward(&input(vec![0.55])).unwrap();
        assert!(
            !g.is_open(0),
            "gate should stay closed within upper hysteresis band"
        );

        // Signal above threshold+hysteresis → opens
        g.forward(&input(vec![0.65])).unwrap();
        assert!(g.is_open(0), "gate should open above threshold+hysteresis");
    }

    // -- Soft gating -------------------------------------------------------

    #[test]
    fn test_soft_gating_sigmoid() {
        let mut g = Gate::with_config(GateConfig {
            num_channels: 1,
            threshold: 0.5,
            soft_gating: true,
            sigmoid_steepness: 10.0,
            gain: 1.0,
            ..default_config()
        });

        let out = g.forward(&input(vec![0.5])).unwrap();
        // At threshold, sigmoid should output 0.5
        assert!(
            (out.gate_values[0] - 0.5).abs() < 1e-10,
            "sigmoid at threshold should be 0.5, got {}",
            out.gate_values[0]
        );
    }

    #[test]
    fn test_soft_gating_above_threshold() {
        let mut g = Gate::with_config(GateConfig {
            num_channels: 1,
            threshold: 0.5,
            soft_gating: true,
            sigmoid_steepness: 10.0,
            gain: 1.0,
            ..default_config()
        });

        let out = g.forward(&input(vec![1.0])).unwrap();
        // Well above threshold → gate should be close to 1.0
        assert!(
            out.gate_values[0] > 0.99,
            "soft gate well above threshold should be ~1.0, got {}",
            out.gate_values[0]
        );
    }

    #[test]
    fn test_soft_gating_below_threshold() {
        let mut g = Gate::with_config(GateConfig {
            num_channels: 1,
            threshold: 0.5,
            soft_gating: true,
            sigmoid_steepness: 10.0,
            gain: 1.0,
            ..default_config()
        });

        let out = g.forward(&input(vec![0.0])).unwrap();
        // Well below threshold → gate should be close to 0.0
        assert!(
            out.gate_values[0] < 0.01,
            "soft gate well below threshold should be ~0.0, got {}",
            out.gate_values[0]
        );
    }

    #[test]
    fn test_soft_gating_steepness() {
        let mut g_flat = Gate::with_config(GateConfig {
            num_channels: 1,
            threshold: 0.5,
            soft_gating: true,
            sigmoid_steepness: 1.0,
            gain: 1.0,
            ..default_config()
        });
        let mut g_steep = Gate::with_config(GateConfig {
            num_channels: 1,
            threshold: 0.5,
            soft_gating: true,
            sigmoid_steepness: 100.0,
            gain: 1.0,
            ..default_config()
        });

        // Signal slightly above threshold
        let out_flat = g_flat.forward(&input(vec![0.6])).unwrap();
        let out_steep = g_steep.forward(&input(vec![0.6])).unwrap();

        // Steeper sigmoid should be closer to 1.0 above threshold
        assert!(
            out_steep.gate_values[0] > out_flat.gate_values[0],
            "steeper sigmoid should give higher gate value above threshold: steep={}, flat={}",
            out_steep.gate_values[0],
            out_flat.gate_values[0]
        );
    }

    #[test]
    fn test_soft_gating_output_scaled() {
        let mut g = Gate::with_config(GateConfig {
            num_channels: 1,
            threshold: 0.5,
            soft_gating: true,
            sigmoid_steepness: 10.0,
            gain: 2.0,
            ..default_config()
        });

        let out = g.forward(&input(vec![1.0])).unwrap();
        // Output = signal * gate_value * gain ≈ 1.0 * ~1.0 * 2.0 ≈ 2.0
        assert!(
            (out.signals[0] - 2.0).abs() < 0.05,
            "expected ~2.0, got {}",
            out.signals[0]
        );
    }

    // -- Adaptive gain -----------------------------------------------------

    #[test]
    fn test_adaptive_gain_adjusts() {
        let mut g = Gate::with_config(GateConfig {
            num_channels: 1,
            threshold: 0.0, // always open
            hysteresis: 0.0,
            gain: 1.0,
            adaptive_gain: true,
            gain_ema_decay: 0.5,
            target_output_level: 1.0,
            max_gain: 100.0,
            min_gain: 0.01,
            ..default_config()
        });

        let initial_gain = g.effective_gain();
        for _ in 0..20 {
            g.forward(&input(vec![0.1])).unwrap(); // low signal
        }

        // Gain should have increased to compensate for low signal
        assert!(
            g.effective_gain() > initial_gain,
            "adaptive gain should increase for low signals: initial={}, current={}",
            initial_gain,
            g.effective_gain()
        );
    }

    #[test]
    fn test_adaptive_gain_clamped() {
        let mut g = Gate::with_config(GateConfig {
            num_channels: 1,
            threshold: 0.0,
            hysteresis: 0.0,
            gain: 1.0,
            adaptive_gain: true,
            gain_ema_decay: 0.1,
            target_output_level: 1000.0, // very high target
            max_gain: 5.0,
            min_gain: 0.5,
            ..default_config()
        });

        for _ in 0..100 {
            g.forward(&input(vec![0.01])).unwrap();
        }

        assert!(
            g.effective_gain() <= 5.0,
            "gain should be clamped at max: {}",
            g.effective_gain()
        );
    }

    // -- Open fraction -----------------------------------------------------

    #[test]
    fn test_open_fraction_all_open() {
        let mut g = Gate::with_config(GateConfig {
            num_channels: 3,
            threshold: 0.0,
            hysteresis: 0.0,
            gain: 1.0,
            ..default_config()
        });
        let out = g.forward(&input(vec![1.0, 1.0, 1.0])).unwrap();
        assert!(
            (out.open_fraction - 1.0).abs() < 1e-10,
            "all channels open → fraction should be 1.0"
        );
    }

    #[test]
    fn test_open_fraction_all_closed() {
        let mut g = Gate::with_config(GateConfig {
            num_channels: 3,
            threshold: 10.0,
            hysteresis: 0.0,
            gain: 1.0,
            ..default_config()
        });
        let out = g.forward(&input(vec![1.0, 1.0, 1.0])).unwrap();
        assert!(
            out.open_fraction.abs() < 1e-10,
            "all channels closed → fraction should be 0.0"
        );
    }

    // -- EMA smoothing -----------------------------------------------------

    #[test]
    fn test_ema_smoothing() {
        let mut g = Gate::with_config(GateConfig {
            num_channels: 1,
            threshold: 0.5,
            hysteresis: 0.0,
            gain: 1.0,
            ema_decay: 0.8,
            ..default_config()
        });

        // First: open
        let out1 = g.forward(&input(vec![1.0])).unwrap();
        assert!((out1.smoothed_open_fraction - 1.0).abs() < 1e-10);

        // Second: closed
        let out2 = g.forward(&input(vec![0.0])).unwrap();
        // EMA: 0.8 * 1.0 + 0.2 * 0.0 = 0.8
        assert!(
            (out2.smoothed_open_fraction - 0.8).abs() < 0.01,
            "expected ~0.8, got {}",
            out2.smoothed_open_fraction
        );
    }

    #[test]
    fn test_ema_disabled_when_zero() {
        let mut g = Gate::with_config(GateConfig {
            num_channels: 1,
            threshold: 0.5,
            hysteresis: 0.0,
            gain: 1.0,
            ema_decay: 0.0,
            ..default_config()
        });
        let out = g.forward(&input(vec![1.0])).unwrap();
        assert!(
            (out.smoothed_open_fraction - out.open_fraction).abs() < 1e-10,
            "with ema_decay=0, smoothed should equal raw"
        );
    }

    // -- Channel stats -----------------------------------------------------

    #[test]
    fn test_channel_stats() {
        let mut g = Gate::with_config(GateConfig {
            num_channels: 2,
            threshold: 0.5,
            hysteresis: 0.0,
            gain: 1.0,
            ..default_config()
        });

        // Channel 0: passes, Channel 1: blocked
        for _ in 0..10 {
            g.forward(&input(vec![1.0, 0.0])).unwrap();
        }

        let cs = g.channel_stats();
        assert_eq!(cs[0].total_signals, 10);
        assert_eq!(cs[0].pass_count, 10);
        assert_eq!(cs[0].block_count, 0);
        assert!((cs[0].duty_cycle - 1.0).abs() < 1e-10);

        assert_eq!(cs[1].total_signals, 10);
        assert_eq!(cs[1].pass_count, 0);
        assert_eq!(cs[1].block_count, 10);
        assert!(cs[1].duty_cycle.abs() < 1e-10);
    }

    // -- Dynamic threshold -------------------------------------------------

    #[test]
    fn test_set_threshold() {
        let mut g = Gate::with_config(GateConfig {
            num_channels: 1,
            threshold: 0.5,
            hysteresis: 0.0,
            gain: 1.0,
            ..default_config()
        });

        // Signal below original threshold → blocked
        let out = g.forward(&input(vec![0.3])).unwrap();
        assert!(out.gate_values[0] < 0.5);

        // Lower threshold
        g.set_threshold(0.2);

        // Same signal now above threshold → passes
        let out = g.forward(&input(vec![0.3])).unwrap();
        assert!(out.gate_values[0] > 0.5);
    }

    // -- Dynamic gain ------------------------------------------------------

    #[test]
    fn test_set_gain() {
        let mut g = Gate::with_config(GateConfig {
            num_channels: 1,
            threshold: 0.0,
            hysteresis: 0.0,
            gain: 1.0,
            max_gain: 10.0,
            min_gain: 0.1,
            ..default_config()
        });

        g.set_gain(3.0);
        let out = g.forward(&input(vec![1.0])).unwrap();
        assert!(
            (out.effective_gain - 3.0).abs() < 1e-10,
            "gain should be 3.0, got {}",
            out.effective_gain
        );
    }

    #[test]
    fn test_set_gain_clamped() {
        let mut g = Gate::with_config(GateConfig {
            max_gain: 5.0,
            min_gain: 0.5,
            ..default_config()
        });

        g.set_gain(100.0);
        assert!(
            (g.effective_gain() - 5.0).abs() < 1e-10,
            "gain should be clamped to max"
        );

        g.set_gain(0.01);
        assert!(
            (g.effective_gain() - 0.5).abs() < 1e-10,
            "gain should be clamped to min"
        );
    }

    // -- Windowed statistics -----------------------------------------------

    #[test]
    fn test_windowed_mean() {
        let mut g = Gate::with_config(GateConfig {
            num_channels: 1,
            threshold: 0.5,
            hysteresis: 0.0,
            gain: 1.0,
            ..default_config()
        });
        // Alternate open and closed
        for i in 0..10 {
            if i % 2 == 0 {
                g.forward(&input(vec![1.0])).unwrap();
            } else {
                g.forward(&input(vec![0.0])).unwrap();
            }
        }
        let mean = g.windowed_mean().unwrap();
        assert!(
            (mean - 0.5).abs() < 0.01,
            "alternating open/closed should give mean ~0.5, got {}",
            mean
        );
    }

    #[test]
    fn test_windowed_mean_empty() {
        let g = default_gate();
        assert!(g.windowed_mean().is_none());
    }

    #[test]
    fn test_windowed_std_insufficient() {
        let mut g = default_gate();
        g.forward(&input(vec![1.0, 1.0, 1.0])).unwrap();
        assert!(g.windowed_std().is_none());
    }

    // -- Stats tracking ----------------------------------------------------

    #[test]
    fn test_stats_tracking() {
        let mut g = default_gate();
        g.forward(&input(vec![1.0, 0.0, 1.0])).unwrap();
        g.forward(&input(vec![0.0, 1.0, 0.0])).unwrap();

        assert_eq!(g.stats().total_operations, 2);
        assert!(g.stats().avg_open_fraction() > 0.0);
    }

    #[test]
    fn test_transition_count() {
        let mut g = Gate::with_config(GateConfig {
            num_channels: 1,
            threshold: 0.5,
            hysteresis: 0.0,
            gain: 1.0,
            ..default_config()
        });

        // Closed → open → closed → open
        g.forward(&input(vec![0.0])).unwrap(); // closed
        g.forward(&input(vec![1.0])).unwrap(); // open (transition)
        g.forward(&input(vec![0.0])).unwrap(); // closed (transition)
        g.forward(&input(vec![1.0])).unwrap(); // open (transition)

        assert_eq!(g.stats().total_transitions, 3, "should have 3 transitions");
    }

    // -- Reset -------------------------------------------------------------

    #[test]
    fn test_reset() {
        let mut g = default_gate();
        for _ in 0..10 {
            g.forward(&input(vec![1.0, 0.5, 0.0])).unwrap();
        }
        assert!(g.stats().total_operations > 0);

        g.reset();

        assert_eq!(g.stats().total_operations, 0);
        assert_eq!(g.stats().total_transitions, 0);
        assert!(g.windowed_mean().is_none());
        assert!(!g.is_open(0));
    }

    // -- Window eviction ---------------------------------------------------

    #[test]
    fn test_window_eviction() {
        let mut g = Gate::with_config(GateConfig {
            window_size: 3,
            ..default_config()
        });
        for _ in 0..10 {
            g.forward(&input(vec![1.0, 0.0, 1.0])).unwrap();
        }
        assert_eq!(g.history.len(), 3);
    }

    // -- Is open accessor --------------------------------------------------

    #[test]
    fn test_is_open_accessor() {
        let mut g = Gate::with_config(GateConfig {
            num_channels: 2,
            threshold: 0.5,
            hysteresis: 0.0,
            gain: 1.0,
            ..default_config()
        });
        g.forward(&input(vec![1.0, 0.0])).unwrap();
        assert!(g.is_open(0));
        assert!(!g.is_open(1));
    }

    #[test]
    fn test_is_open_out_of_bounds() {
        let g = default_gate();
        assert!(!g.is_open(100)); // should not panic
    }

    // -- Num channels accessor ---------------------------------------------

    #[test]
    fn test_num_channels() {
        let g = Gate::with_config(GateConfig {
            num_channels: 5,
            ..default_config()
        });
        assert_eq!(g.num_channels(), 5);
    }

    // -- Negative signals --------------------------------------------------

    #[test]
    fn test_negative_signal_gating() {
        let mut g = Gate::with_config(GateConfig {
            num_channels: 1,
            threshold: -0.5,
            hysteresis: 0.0,
            gain: 1.0,
            ..default_config()
        });
        // Negative signal above negative threshold
        let out = g.forward(&input(vec![-0.3])).unwrap();
        assert!(
            out.gate_values[0] > 0.5,
            "signal above threshold should pass"
        );
        assert!(
            (out.signals[0] - (-0.3)).abs() < 1e-10,
            "output should match input when gate is open"
        );
    }

    // -- Stats defaults ----------------------------------------------------

    #[test]
    fn test_stats_defaults() {
        let g = default_gate();
        assert_eq!(g.stats().total_operations, 0);
        assert_eq!(g.stats().total_transitions, 0);
        assert_eq!(g.stats().avg_open_fraction(), 0.0);
    }
}
