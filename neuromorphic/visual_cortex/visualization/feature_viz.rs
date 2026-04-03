//! Feature Visualization
//!
//! Part of the Visual Cortex region — Visualization component.
//!
//! Implements feature visualization techniques for understanding what patterns
//! the visual cortex has learned to detect in market data:
//!
//! - **Activation maximization**: Generate synthetic inputs that maximally
//!   activate a given channel/neuron, revealing learned features.
//! - **Channel importance ranking**: Score channels by their mean activation,
//!   variance, or max activation to identify the most informative features.
//! - **Top-K retrieval**: Given a set of recorded activations, retrieve the
//!   input samples that most strongly activate a target channel.
//!
//! # Trading context
//!
//! - Channels may correspond to learned price patterns (double-top, head-and-shoulders),
//!   volume anomalies, volatility regimes, etc.
//! - Feature visualization helps interpretability: understanding *why* the model
//!   signals a particular trade.
//! - Channel importance ranking guides feature engineering and model pruning.

use crate::common::{Error, Result};
use std::collections::{BTreeMap, HashMap, VecDeque};

// ─── Configuration ──────────────────────────────────────────────────────────

/// Configuration for the feature visualization module.
#[derive(Debug, Clone)]
pub struct FeatureVizConfig {
    /// Maximum number of channels that can be registered.
    pub max_channels: usize,

    /// Number of gradient-ascent steps for activation maximization.
    pub max_steps: usize,

    /// Learning rate for activation maximization gradient ascent.
    pub learning_rate: f64,

    /// L2 regularization weight for activation maximization.
    pub l2_decay: f64,

    /// Default number of top-K results to return.
    pub default_top_k: usize,

    /// Maximum number of activation records to keep per channel (ring buffer).
    pub max_records_per_channel: usize,

    /// EMA decay factor for quality tracking. Must be in (0, 1).
    pub ema_decay: f64,

    /// Minimum samples before EMA is considered warmed up.
    pub min_samples: usize,

    /// Rolling window size for windowed diagnostics.
    pub window_size: usize,
}

impl Default for FeatureVizConfig {
    fn default() -> Self {
        Self {
            max_channels: 512,
            max_steps: 100,
            learning_rate: 0.01,
            l2_decay: 1e-4,
            default_top_k: 5,
            max_records_per_channel: 1000,
            ema_decay: 0.05,
            min_samples: 10,
            window_size: 100,
        }
    }
}

// ─── Channel importance metrics ─────────────────────────────────────────────

/// How to rank channels by importance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImportanceMetric {
    /// Rank by mean activation value.
    MeanActivation,
    /// Rank by activation variance (higher variance = more informative).
    Variance,
    /// Rank by maximum observed activation.
    MaxActivation,
    /// Rank by activation range (max - min).
    Range,
}

impl Default for ImportanceMetric {
    fn default() -> Self {
        Self::MeanActivation
    }
}

impl ImportanceMetric {
    /// Human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            Self::MeanActivation => "mean_activation",
            Self::Variance => "variance",
            Self::MaxActivation => "max_activation",
            Self::Range => "range",
        }
    }
}

// ─── Channel state ──────────────────────────────────────────────────────────

/// Running statistics for a single channel.
#[derive(Debug, Clone)]
struct ChannelState {
    /// Channel name / identifier.
    name: String,

    /// Number of activations recorded.
    count: u64,

    /// Running mean (Welford).
    mean: f64,

    /// Running M2 for variance (Welford).
    m2: f64,

    /// Minimum observed activation.
    min: f64,

    /// Maximum observed activation.
    max: f64,

    /// Ring buffer of recent (activation, sample_id) pairs for top-K retrieval.
    records: VecDeque<ActivationRecord>,

    /// Maximum records to keep.
    capacity: usize,
}

/// A single activation observation.
#[derive(Debug, Clone)]
struct ActivationRecord {
    /// The activation value.
    activation: f64,

    /// An opaque identifier for the input sample that produced this activation.
    sample_id: u64,
}

impl ChannelState {
    fn new(name: String, capacity: usize) -> Self {
        Self {
            name,
            count: 0,
            mean: 0.0,
            m2: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            records: VecDeque::with_capacity(capacity.min(1024)),
            capacity,
        }
    }

    fn record(&mut self, activation: f64, sample_id: u64) {
        self.count += 1;
        let delta = activation - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = activation - self.mean;
        self.m2 += delta * delta2;

        if activation < self.min {
            self.min = activation;
        }
        if activation > self.max {
            self.max = activation;
        }

        if self.records.len() >= self.capacity {
            self.records.pop_front();
        }
        self.records.push_back(ActivationRecord {
            activation,
            sample_id,
        });
    }

    fn variance(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        self.m2 / (self.count - 1) as f64
    }

    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    fn range(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        self.max - self.min
    }

    fn importance(&self, metric: ImportanceMetric) -> f64 {
        match metric {
            ImportanceMetric::MeanActivation => self.mean,
            ImportanceMetric::Variance => self.variance(),
            ImportanceMetric::MaxActivation => {
                if self.count == 0 {
                    f64::NEG_INFINITY
                } else {
                    self.max
                }
            }
            ImportanceMetric::Range => self.range(),
        }
    }

    /// Return the top-K records by activation value (descending).
    fn top_k(&self, k: usize) -> Vec<TopKEntry> {
        let mut sorted: Vec<_> = self
            .records
            .iter()
            .map(|r| TopKEntry {
                sample_id: r.sample_id,
                activation: r.activation,
            })
            .collect();
        sorted.sort_by(|a, b| {
            b.activation
                .partial_cmp(&a.activation)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.truncate(k);
        sorted
    }

    fn reset_stats(&mut self) {
        self.count = 0;
        self.mean = 0.0;
        self.m2 = 0.0;
        self.min = f64::INFINITY;
        self.max = f64::NEG_INFINITY;
        self.records.clear();
    }
}

// ─── Public output types ────────────────────────────────────────────────────

/// A single entry in a top-K result set.
#[derive(Debug, Clone)]
pub struct TopKEntry {
    /// The sample ID that produced this activation.
    pub sample_id: u64,

    /// The activation value.
    pub activation: f64,
}

/// Channel importance ranking entry.
#[derive(Debug, Clone)]
pub struct ChannelRanking {
    /// Channel name.
    pub channel: String,

    /// Importance score according to the chosen metric.
    pub score: f64,

    /// Rank (1-based, 1 = most important).
    pub rank: usize,
}

/// Result of activation maximization.
#[derive(Debug, Clone)]
pub struct MaximizationResult {
    /// The channel that was targeted.
    pub channel: String,

    /// The synthesized input that maximally activates the channel.
    pub optimized_input: Vec<f64>,

    /// Final activation value achieved.
    pub final_activation: f64,

    /// Number of optimization steps taken.
    pub steps_taken: usize,

    /// Activation history over the optimization (one per step).
    pub activation_history: Vec<f64>,

    /// Whether the optimization converged (activation stopped increasing).
    pub converged: bool,
}

/// Summary statistics for a single channel.
#[derive(Debug, Clone)]
pub struct ChannelSummary {
    /// Channel name.
    pub name: String,

    /// Number of activations recorded.
    pub count: u64,

    /// Mean activation.
    pub mean: f64,

    /// Standard deviation of activations.
    pub std_dev: f64,

    /// Minimum activation.
    pub min: f64,

    /// Maximum activation.
    pub max: f64,

    /// Activation range.
    pub range: f64,
}

/// Windowed diagnostic record.
#[derive(Debug, Clone)]
struct WindowRecord {
    mean_importance: f64,
    channels_updated: usize,
    tick: u64,
}

/// Global statistics for the feature visualization module.
#[derive(Debug, Clone, Default)]
pub struct FeatureVizStats {
    /// Total number of activation records ingested.
    pub total_records: u64,

    /// Total number of maximization runs performed.
    pub total_maximizations: u64,

    /// Total number of ranking queries performed.
    pub total_rankings: u64,

    /// Total number of top-K queries performed.
    pub total_top_k_queries: u64,

    /// Number of registered channels.
    pub registered_channels: usize,

    /// Peak mean importance observed.
    pub peak_mean_importance: f64,

    /// Sum of mean importance for averaging.
    sum_mean_importance: f64,

    /// Count of importance observations.
    importance_observations: u64,
}

impl FeatureVizStats {
    /// Mean of the per-tick mean-importance values observed.
    pub fn overall_mean_importance(&self) -> f64 {
        if self.importance_observations == 0 {
            return 0.0;
        }
        self.sum_mean_importance / self.importance_observations as f64
    }
}

// ─── Activation function for maximization ───────────────────────────────────

/// A user-supplied (or default) activation function used during maximization.
///
/// Given an input vector, returns the scalar activation for the target channel.
pub type ActivationFn = Box<dyn Fn(&[f64]) -> f64 + Send + Sync>;

/// Numerical gradient of an activation function w.r.t. the input.
fn numerical_gradient(f: &ActivationFn, input: &[f64], epsilon: f64) -> Vec<f64> {
    let n = input.len();
    let mut grad = vec![0.0; n];
    let mut perturbed = input.to_vec();
    for i in 0..n {
        let orig = perturbed[i];
        perturbed[i] = orig + epsilon;
        let f_plus = f(&perturbed);
        perturbed[i] = orig - epsilon;
        let f_minus = f(&perturbed);
        perturbed[i] = orig;
        grad[i] = (f_plus - f_minus) / (2.0 * epsilon);
    }
    grad
}

// ─── Main struct ────────────────────────────────────────────────────────────

/// Feature visualization module.
///
/// Provides activation maximization, channel importance ranking, and top-K
/// sample retrieval for understanding learned features in the visual cortex.
pub struct FeatureViz {
    config: FeatureVizConfig,

    /// Registered channels, keyed by name.
    channels: HashMap<String, ChannelState>,

    /// Insertion-order tracking for deterministic iteration.
    channel_order: Vec<String>,

    /// Default importance metric for ranking.
    importance_metric: ImportanceMetric,

    /// EMA of the mean importance across all channels.
    ema_importance: f64,
    ema_initialized: bool,

    /// Rolling window.
    recent: VecDeque<WindowRecord>,

    /// Tick counter.
    current_tick: u64,

    /// Running statistics.
    stats: FeatureVizStats,
}

impl Default for FeatureViz {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureViz {
    /// Create an instance with default configuration.
    pub fn new() -> Self {
        Self::with_config(FeatureVizConfig::default()).expect("default config must be valid")
    }

    /// Create an instance from a validated configuration.
    pub fn with_config(config: FeatureVizConfig) -> Result<Self> {
        if config.max_channels == 0 {
            return Err(Error::InvalidInput("max_channels must be > 0".into()));
        }
        if config.max_steps == 0 {
            return Err(Error::InvalidInput("max_steps must be > 0".into()));
        }
        if config.learning_rate <= 0.0 {
            return Err(Error::InvalidInput("learning_rate must be > 0".into()));
        }
        if config.l2_decay < 0.0 {
            return Err(Error::InvalidInput("l2_decay must be >= 0".into()));
        }
        if config.default_top_k == 0 {
            return Err(Error::InvalidInput("default_top_k must be > 0".into()));
        }
        if config.max_records_per_channel == 0 {
            return Err(Error::InvalidInput(
                "max_records_per_channel must be > 0".into(),
            ));
        }
        if config.ema_decay <= 0.0 || config.ema_decay >= 1.0 {
            return Err(Error::InvalidInput("ema_decay must be in (0, 1)".into()));
        }
        if config.window_size == 0 {
            return Err(Error::InvalidInput("window_size must be > 0".into()));
        }

        Ok(Self {
            config,
            channels: HashMap::new(),
            channel_order: Vec::new(),
            importance_metric: ImportanceMetric::default(),
            ema_importance: 0.0,
            ema_initialized: false,
            recent: VecDeque::new(),
            current_tick: 0,
            stats: FeatureVizStats::default(),
        })
    }

    // ── Channel registration ────────────────────────────────────────────

    /// Register a new channel for tracking.
    pub fn register_channel(&mut self, name: &str) -> Result<()> {
        if self.channels.contains_key(name) {
            return Err(Error::InvalidInput(format!(
                "channel '{}' already registered",
                name
            )));
        }
        if self.channels.len() >= self.config.max_channels {
            return Err(Error::InvalidInput(format!(
                "max channels ({}) reached",
                self.config.max_channels
            )));
        }
        self.channels.insert(
            name.to_string(),
            ChannelState::new(name.to_string(), self.config.max_records_per_channel),
        );
        self.channel_order.push(name.to_string());
        self.stats.registered_channels = self.channels.len();
        Ok(())
    }

    /// Deregister a channel.
    pub fn deregister_channel(&mut self, name: &str) -> Result<()> {
        if self.channels.remove(name).is_none() {
            return Err(Error::InvalidInput(format!("channel '{}' not found", name)));
        }
        self.channel_order.retain(|n| n != name);
        self.stats.registered_channels = self.channels.len();
        Ok(())
    }

    /// Number of registered channels.
    pub fn channel_count(&self) -> usize {
        self.channels.len()
    }

    /// Names of registered channels in registration order.
    pub fn channel_names(&self) -> &[String] {
        &self.channel_order
    }

    // ── Recording activations ───────────────────────────────────────────

    /// Record an activation value for a channel from a specific sample.
    pub fn record_activation(
        &mut self,
        channel: &str,
        activation: f64,
        sample_id: u64,
    ) -> Result<()> {
        let ch = self
            .channels
            .get_mut(channel)
            .ok_or_else(|| Error::InvalidInput(format!("channel '{}' not found", channel)))?;
        ch.record(activation, sample_id);
        self.stats.total_records += 1;
        Ok(())
    }

    /// Record activations for multiple channels from the same sample.
    ///
    /// `activations` maps channel_name → activation_value.
    pub fn record_batch(&mut self, activations: &[(&str, f64)], sample_id: u64) -> Result<usize> {
        let mut recorded = 0;
        for &(name, value) in activations {
            if let Some(ch) = self.channels.get_mut(name) {
                ch.record(value, sample_id);
                self.stats.total_records += 1;
                recorded += 1;
            }
        }
        Ok(recorded)
    }

    // ── Channel summary ─────────────────────────────────────────────────

    /// Get a summary of a specific channel's statistics.
    pub fn channel_summary(&self, name: &str) -> Option<ChannelSummary> {
        self.channels.get(name).map(|ch| ChannelSummary {
            name: ch.name.clone(),
            count: ch.count,
            mean: ch.mean,
            std_dev: ch.std_dev(),
            min: if ch.count == 0 { 0.0 } else { ch.min },
            max: if ch.count == 0 { 0.0 } else { ch.max },
            range: ch.range(),
        })
    }

    /// Get summaries for all channels, in registration order.
    pub fn all_summaries(&self) -> Vec<ChannelSummary> {
        self.channel_order
            .iter()
            .filter_map(|name| self.channel_summary(name))
            .collect()
    }

    // ── Importance ranking ──────────────────────────────────────────────

    /// Set the default importance metric for ranking.
    pub fn set_importance_metric(&mut self, metric: ImportanceMetric) {
        self.importance_metric = metric;
    }

    /// Current importance metric.
    pub fn importance_metric(&self) -> ImportanceMetric {
        self.importance_metric
    }

    /// Rank all channels by importance using the current metric.
    ///
    /// Returns a list sorted by score descending (most important first).
    pub fn rank_channels(&mut self) -> Vec<ChannelRanking> {
        self.rank_channels_by(self.importance_metric)
    }

    /// Rank all channels by a specific metric.
    pub fn rank_channels_by(&mut self, metric: ImportanceMetric) -> Vec<ChannelRanking> {
        self.stats.total_rankings += 1;

        let mut rankings: Vec<(String, f64)> = self
            .channel_order
            .iter()
            .filter_map(|name| {
                self.channels
                    .get(name)
                    .map(|ch| (name.clone(), ch.importance(metric)))
            })
            .collect();

        rankings.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        rankings
            .into_iter()
            .enumerate()
            .map(|(i, (channel, score))| ChannelRanking {
                channel,
                score,
                rank: i + 1,
            })
            .collect()
    }

    /// Get the importance score for a single channel.
    pub fn channel_importance(&self, name: &str) -> Option<f64> {
        self.channels
            .get(name)
            .map(|ch| ch.importance(self.importance_metric))
    }

    /// Get the importance score for a single channel with a specific metric.
    pub fn channel_importance_by(&self, name: &str, metric: ImportanceMetric) -> Option<f64> {
        self.channels.get(name).map(|ch| ch.importance(metric))
    }

    // ── Top-K retrieval ─────────────────────────────────────────────────

    /// Return the top-K samples that most strongly activated a channel.
    ///
    /// Uses the configured `default_top_k` value.
    pub fn top_k(&mut self, channel: &str) -> Result<Vec<TopKEntry>> {
        self.top_k_n(channel, self.config.default_top_k)
    }

    /// Return the top-K samples with a custom K value.
    pub fn top_k_n(&mut self, channel: &str, k: usize) -> Result<Vec<TopKEntry>> {
        self.stats.total_top_k_queries += 1;
        let ch = self
            .channels
            .get(channel)
            .ok_or_else(|| Error::InvalidInput(format!("channel '{}' not found", channel)))?;
        Ok(ch.top_k(k))
    }

    // ── Activation maximization ─────────────────────────────────────────

    /// Run activation maximization for a channel.
    ///
    /// `activation_fn` should compute the scalar activation of the target
    /// channel given an input vector.
    ///
    /// `input_dim` is the length of the input vector.
    ///
    /// Returns a `MaximizationResult` containing the optimized synthetic input.
    pub fn maximize(
        &mut self,
        channel: &str,
        activation_fn: &ActivationFn,
        input_dim: usize,
    ) -> Result<MaximizationResult> {
        self.maximize_from(channel, activation_fn, &vec![0.0; input_dim])
    }

    /// Run activation maximization starting from a given initial input.
    pub fn maximize_from(
        &mut self,
        channel: &str,
        activation_fn: &ActivationFn,
        initial_input: &[f64],
    ) -> Result<MaximizationResult> {
        if !self.channels.contains_key(channel) {
            return Err(Error::InvalidInput(format!(
                "channel '{}' not found",
                channel
            )));
        }
        if initial_input.is_empty() {
            return Err(Error::InvalidInput("input_dim must be > 0".into()));
        }

        self.stats.total_maximizations += 1;

        let lr = self.config.learning_rate;
        let l2 = self.config.l2_decay;
        let epsilon = 1e-5;
        let convergence_threshold = 1e-8;
        let max_steps = self.config.max_steps;

        let mut input = initial_input.to_vec();
        let mut activation_history = Vec::with_capacity(max_steps);
        let mut converged = false;
        let mut prev_activation = f64::NEG_INFINITY;

        for step in 0..max_steps {
            let current_activation = activation_fn(&input);
            activation_history.push(current_activation);

            // Check convergence
            if step > 0 && (current_activation - prev_activation).abs() < convergence_threshold {
                converged = true;
                break;
            }
            prev_activation = current_activation;

            // Compute gradient via finite differences
            let grad = numerical_gradient(activation_fn, &input, epsilon);

            // Gradient ascent step with L2 regularization
            for i in 0..input.len() {
                input[i] += lr * grad[i] - lr * l2 * input[i];
            }
        }

        let final_activation = activation_fn(&input);
        if activation_history.is_empty() || *activation_history.last().unwrap() != final_activation
        {
            activation_history.push(final_activation);
        }

        Ok(MaximizationResult {
            channel: channel.to_string(),
            optimized_input: input,
            final_activation,
            steps_taken: activation_history.len(),
            activation_history,
            converged,
        })
    }

    // ── Tick & EMA ──────────────────────────────────────────────────────

    /// Advance the tick counter, update EMA and windowed diagnostics.
    pub fn tick(&mut self) {
        self.current_tick += 1;

        // Compute mean importance across all channels
        let (sum, count) = self
            .channel_order
            .iter()
            .fold((0.0, 0usize), |(s, c), name| {
                if let Some(ch) = self.channels.get(name) {
                    if ch.count > 0 {
                        (s + ch.importance(self.importance_metric), c + 1)
                    } else {
                        (s, c)
                    }
                } else {
                    (s, c)
                }
            });

        let mean_importance = if count > 0 { sum / count as f64 } else { 0.0 };

        // Update EMA
        let alpha = self.config.ema_decay;
        if !self.ema_initialized {
            self.ema_importance = mean_importance;
            self.ema_initialized = true;
        } else {
            self.ema_importance = alpha * mean_importance + (1.0 - alpha) * self.ema_importance;
        }

        // Update stats
        self.stats.sum_mean_importance += mean_importance;
        self.stats.importance_observations += 1;
        if mean_importance > self.stats.peak_mean_importance {
            self.stats.peak_mean_importance = mean_importance;
        }

        // Update window
        if self.recent.len() >= self.config.window_size {
            self.recent.pop_front();
        }
        self.recent.push_back(WindowRecord {
            mean_importance,
            channels_updated: count,
            tick: self.current_tick,
        });
    }

    // ── Accessors ───────────────────────────────────────────────────────

    /// Current EMA of mean channel importance.
    pub fn ema_importance(&self) -> f64 {
        self.ema_importance
    }

    /// Whether the EMA has received enough observations.
    pub fn is_warmed_up(&self) -> bool {
        self.stats.importance_observations as usize >= self.config.min_samples
    }

    /// Current tick counter.
    pub fn current_tick(&self) -> u64 {
        self.current_tick
    }

    /// Reference to running statistics.
    pub fn stats(&self) -> &FeatureVizStats {
        &self.stats
    }

    /// Confidence score in [0, 1] based on observation count vs min_samples.
    pub fn confidence(&self) -> f64 {
        let n = self.stats.importance_observations as f64;
        let min = self.config.min_samples as f64;
        (n / min).min(1.0)
    }

    // ── Windowed diagnostics ────────────────────────────────────────────

    /// Windowed mean importance across the rolling window.
    pub fn windowed_mean_importance(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.recent.iter().map(|r| r.mean_importance).sum();
        sum / self.recent.len() as f64
    }

    /// Windowed mean of the number of channels updated per tick.
    pub fn windowed_mean_channels_updated(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.recent.iter().map(|r| r.channels_updated as f64).sum();
        sum / self.recent.len() as f64
    }

    /// Returns `true` if importance has been trending upward over the window.
    pub fn is_importance_increasing(&self) -> bool {
        if self.recent.len() < 4 {
            return false;
        }
        let mid = self.recent.len() / 2;
        let first: f64 = self
            .recent
            .iter()
            .take(mid)
            .map(|r| r.mean_importance)
            .sum::<f64>()
            / mid as f64;
        let second: f64 = self
            .recent
            .iter()
            .skip(mid)
            .map(|r| r.mean_importance)
            .sum::<f64>()
            / (self.recent.len() - mid) as f64;
        second > first
    }

    /// Returns `true` if importance has been trending downward over the window.
    pub fn is_importance_decreasing(&self) -> bool {
        if self.recent.len() < 4 {
            return false;
        }
        let mid = self.recent.len() / 2;
        let first: f64 = self
            .recent
            .iter()
            .take(mid)
            .map(|r| r.mean_importance)
            .sum::<f64>()
            / mid as f64;
        let second: f64 = self
            .recent
            .iter()
            .skip(mid)
            .map(|r| r.mean_importance)
            .sum::<f64>()
            / (self.recent.len() - mid) as f64;
        second < first
    }

    // ── Reset ───────────────────────────────────────────────────────────

    /// Reset EMA, window, and per-channel stats but keep channel registrations.
    pub fn reset(&mut self) {
        self.ema_importance = 0.0;
        self.ema_initialized = false;
        self.recent.clear();
        self.current_tick = 0;
        self.stats = FeatureVizStats {
            registered_channels: self.channels.len(),
            ..Default::default()
        };
        for ch in self.channels.values_mut() {
            ch.reset_stats();
        }
    }

    /// Full reset: deregister all channels and reset everything.
    pub fn reset_all(&mut self) {
        self.channels.clear();
        self.channel_order.clear();
        self.ema_importance = 0.0;
        self.ema_initialized = false;
        self.recent.clear();
        self.current_tick = 0;
        self.stats = FeatureVizStats::default();
    }

    /// Main processing function — no-op for trait compatibility.
    pub fn process(&self) -> Result<()> {
        Ok(())
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helpers -----------------------------------------------------------------

    fn small_config() -> FeatureVizConfig {
        FeatureVizConfig {
            max_channels: 10,
            max_steps: 50,
            learning_rate: 0.1,
            l2_decay: 0.001,
            default_top_k: 3,
            max_records_per_channel: 20,
            ema_decay: 0.1,
            min_samples: 3,
            window_size: 10,
        }
    }

    fn small_fv() -> FeatureViz {
        FeatureViz::with_config(small_config()).unwrap()
    }

    fn with_channels(names: &[&str]) -> FeatureViz {
        let mut fv = small_fv();
        for name in names {
            fv.register_channel(name).unwrap();
        }
        fv
    }

    // Construction ------------------------------------------------------------

    #[test]
    fn test_basic() {
        let fv = FeatureViz::new();
        assert!(fv.process().is_ok());
    }

    #[test]
    fn test_default() {
        let fv = FeatureViz::new();
        assert_eq!(fv.channel_count(), 0);
        assert_eq!(fv.current_tick(), 0);
    }

    #[test]
    fn test_with_config() {
        let cfg = small_config();
        let fv = FeatureViz::with_config(cfg).unwrap();
        assert_eq!(fv.channel_count(), 0);
    }

    #[test]
    fn test_invalid_config_max_channels() {
        let mut cfg = small_config();
        cfg.max_channels = 0;
        assert!(FeatureViz::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_max_steps() {
        let mut cfg = small_config();
        cfg.max_steps = 0;
        assert!(FeatureViz::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_learning_rate() {
        let mut cfg = small_config();
        cfg.learning_rate = 0.0;
        assert!(FeatureViz::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_l2_decay_negative() {
        let mut cfg = small_config();
        cfg.l2_decay = -0.01;
        assert!(FeatureViz::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_default_top_k() {
        let mut cfg = small_config();
        cfg.default_top_k = 0;
        assert!(FeatureViz::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_max_records() {
        let mut cfg = small_config();
        cfg.max_records_per_channel = 0;
        assert!(FeatureViz::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_ema_decay_zero() {
        let mut cfg = small_config();
        cfg.ema_decay = 0.0;
        assert!(FeatureViz::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_ema_decay_one() {
        let mut cfg = small_config();
        cfg.ema_decay = 1.0;
        assert!(FeatureViz::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_window_size() {
        let mut cfg = small_config();
        cfg.window_size = 0;
        assert!(FeatureViz::with_config(cfg).is_err());
    }

    // Channel registration ----------------------------------------------------

    #[test]
    fn test_register_channel() {
        let mut fv = small_fv();
        fv.register_channel("ch_a").unwrap();
        assert_eq!(fv.channel_count(), 1);
        assert_eq!(fv.channel_names(), &["ch_a"]);
    }

    #[test]
    fn test_register_duplicate() {
        let mut fv = small_fv();
        fv.register_channel("ch_a").unwrap();
        assert!(fv.register_channel("ch_a").is_err());
    }

    #[test]
    fn test_register_max_capacity() {
        let mut fv = small_fv(); // max_channels = 10
        for i in 0..10 {
            fv.register_channel(&format!("ch_{}", i)).unwrap();
        }
        assert!(fv.register_channel("overflow").is_err());
    }

    #[test]
    fn test_deregister_channel() {
        let mut fv = with_channels(&["ch_a", "ch_b"]);
        fv.deregister_channel("ch_a").unwrap();
        assert_eq!(fv.channel_count(), 1);
        assert_eq!(fv.channel_names(), &["ch_b"]);
    }

    #[test]
    fn test_deregister_nonexistent() {
        let mut fv = small_fv();
        assert!(fv.deregister_channel("nope").is_err());
    }

    #[test]
    fn test_channel_order_preserved() {
        let fv = with_channels(&["c", "a", "b"]);
        assert_eq!(fv.channel_names(), &["c", "a", "b"]);
    }

    #[test]
    fn test_deregister_preserves_order() {
        let mut fv = with_channels(&["c", "a", "b"]);
        fv.deregister_channel("a").unwrap();
        assert_eq!(fv.channel_names(), &["c", "b"]);
    }

    // Recording activations ---------------------------------------------------

    #[test]
    fn test_record_activation() {
        let mut fv = with_channels(&["ch"]);
        fv.record_activation("ch", 1.5, 100).unwrap();
        fv.record_activation("ch", 2.5, 101).unwrap();
        let summary = fv.channel_summary("ch").unwrap();
        assert_eq!(summary.count, 2);
        assert!((summary.mean - 2.0).abs() < 1e-10);
        assert!((summary.min - 1.5).abs() < 1e-10);
        assert!((summary.max - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_record_nonexistent_channel() {
        let mut fv = small_fv();
        assert!(fv.record_activation("nope", 1.0, 1).is_err());
    }

    #[test]
    fn test_record_batch() {
        let mut fv = with_channels(&["a", "b", "c"]);
        let recorded = fv
            .record_batch(&[("a", 1.0), ("b", 2.0), ("nonexistent", 3.0)], 42)
            .unwrap();
        assert_eq!(recorded, 2); // "nonexistent" is silently skipped
        assert_eq!(fv.channel_summary("a").unwrap().count, 1);
        assert_eq!(fv.channel_summary("b").unwrap().count, 1);
        assert_eq!(fv.channel_summary("c").unwrap().count, 0);
    }

    #[test]
    fn test_record_stats_updated() {
        let mut fv = with_channels(&["ch"]);
        fv.record_activation("ch", 1.0, 1).unwrap();
        fv.record_activation("ch", 2.0, 2).unwrap();
        assert_eq!(fv.stats().total_records, 2);
    }

    #[test]
    fn test_ring_buffer_eviction() {
        let mut fv = small_fv(); // max_records = 20
        fv.register_channel("ch").unwrap();
        for i in 0..30 {
            fv.record_activation("ch", i as f64, i).unwrap();
        }
        // top_k should only see records from the last 20
        let top = fv.top_k_n("ch", 5).unwrap();
        assert_eq!(top.len(), 5);
        // Highest should be 29
        assert!((top[0].activation - 29.0).abs() < 1e-10);
        // 5th highest should be 25
        assert!((top[4].activation - 25.0).abs() < 1e-10);
    }

    // Channel summary ---------------------------------------------------------

    #[test]
    fn test_channel_summary_empty() {
        let fv = with_channels(&["ch"]);
        let s = fv.channel_summary("ch").unwrap();
        assert_eq!(s.count, 0);
        assert_eq!(s.mean, 0.0);
        assert_eq!(s.std_dev, 0.0);
        assert_eq!(s.min, 0.0);
        assert_eq!(s.max, 0.0);
        assert_eq!(s.range, 0.0);
    }

    #[test]
    fn test_channel_summary_nonexistent() {
        let fv = small_fv();
        assert!(fv.channel_summary("nope").is_none());
    }

    #[test]
    fn test_channel_summary_std_dev() {
        let mut fv = with_channels(&["ch"]);
        // Known values: [2, 4, 4, 4, 5, 5, 7, 9] → mean=5, variance≈4.571
        for v in &[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
            fv.record_activation("ch", *v, 0).unwrap();
        }
        let s = fv.channel_summary("ch").unwrap();
        assert!((s.mean - 5.0).abs() < 1e-10);
        // Sample variance = 32/7 ≈ 4.571
        let expected_var = 32.0 / 7.0;
        let actual_var = s.std_dev * s.std_dev;
        assert!(
            (actual_var - expected_var).abs() < 1e-6,
            "expected variance ≈ {}, got {}",
            expected_var,
            actual_var
        );
    }

    #[test]
    fn test_all_summaries() {
        let mut fv = with_channels(&["a", "b"]);
        fv.record_activation("a", 1.0, 0).unwrap();
        fv.record_activation("b", 2.0, 1).unwrap();
        let summaries = fv.all_summaries();
        assert_eq!(summaries.len(), 2);
        assert_eq!(summaries[0].name, "a");
        assert_eq!(summaries[1].name, "b");
    }

    // Importance ranking ------------------------------------------------------

    #[test]
    fn test_rank_channels_mean() {
        let mut fv = with_channels(&["low", "high", "mid"]);
        fv.record_activation("low", 1.0, 0).unwrap();
        fv.record_activation("high", 10.0, 1).unwrap();
        fv.record_activation("mid", 5.0, 2).unwrap();
        fv.set_importance_metric(ImportanceMetric::MeanActivation);
        let rankings = fv.rank_channels();
        assert_eq!(rankings[0].channel, "high");
        assert_eq!(rankings[0].rank, 1);
        assert_eq!(rankings[1].channel, "mid");
        assert_eq!(rankings[2].channel, "low");
    }

    #[test]
    fn test_rank_channels_variance() {
        let mut fv = with_channels(&["stable", "volatile"]);
        // Stable: [5, 5, 5] → variance = 0
        for _ in 0..3 {
            fv.record_activation("stable", 5.0, 0).unwrap();
        }
        // Volatile: [1, 10, 1] → variance ≈ 27
        for v in &[1.0, 10.0, 1.0] {
            fv.record_activation("volatile", *v, 0).unwrap();
        }
        let rankings = fv.rank_channels_by(ImportanceMetric::Variance);
        assert_eq!(rankings[0].channel, "volatile");
        assert_eq!(rankings[1].channel, "stable");
    }

    #[test]
    fn test_rank_channels_max() {
        let mut fv = with_channels(&["a", "b"]);
        fv.record_activation("a", 100.0, 0).unwrap();
        fv.record_activation("b", 50.0, 0).unwrap();
        let rankings = fv.rank_channels_by(ImportanceMetric::MaxActivation);
        assert_eq!(rankings[0].channel, "a");
    }

    #[test]
    fn test_rank_channels_range() {
        let mut fv = with_channels(&["narrow", "wide"]);
        fv.record_activation("narrow", 4.0, 0).unwrap();
        fv.record_activation("narrow", 6.0, 1).unwrap();
        fv.record_activation("wide", 0.0, 0).unwrap();
        fv.record_activation("wide", 100.0, 1).unwrap();
        let rankings = fv.rank_channels_by(ImportanceMetric::Range);
        assert_eq!(rankings[0].channel, "wide");
        assert!((rankings[0].score - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_ranking_stats_incremented() {
        let mut fv = with_channels(&["ch"]);
        fv.record_activation("ch", 1.0, 0).unwrap();
        fv.rank_channels();
        fv.rank_channels();
        assert_eq!(fv.stats().total_rankings, 2);
    }

    #[test]
    fn test_channel_importance() {
        let mut fv = with_channels(&["ch"]);
        fv.record_activation("ch", 3.0, 0).unwrap();
        fv.record_activation("ch", 5.0, 1).unwrap();
        let imp = fv.channel_importance("ch").unwrap();
        assert!((imp - 4.0).abs() < 1e-10); // mean
    }

    #[test]
    fn test_channel_importance_nonexistent() {
        let fv = small_fv();
        assert!(fv.channel_importance("nope").is_none());
    }

    #[test]
    fn test_importance_metric_label() {
        assert_eq!(ImportanceMetric::MeanActivation.label(), "mean_activation");
        assert_eq!(ImportanceMetric::Variance.label(), "variance");
        assert_eq!(ImportanceMetric::MaxActivation.label(), "max_activation");
        assert_eq!(ImportanceMetric::Range.label(), "range");
    }

    // Top-K retrieval ---------------------------------------------------------

    #[test]
    fn test_top_k() {
        let mut fv = with_channels(&["ch"]);
        for i in 0..10 {
            fv.record_activation("ch", i as f64, i).unwrap();
        }
        let top = fv.top_k("ch").unwrap(); // default_top_k = 3
        assert_eq!(top.len(), 3);
        assert!((top[0].activation - 9.0).abs() < 1e-10);
        assert_eq!(top[0].sample_id, 9);
        assert!((top[1].activation - 8.0).abs() < 1e-10);
        assert!((top[2].activation - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_top_k_custom() {
        let mut fv = with_channels(&["ch"]);
        for i in 0..5 {
            fv.record_activation("ch", i as f64, i).unwrap();
        }
        let top = fv.top_k_n("ch", 2).unwrap();
        assert_eq!(top.len(), 2);
    }

    #[test]
    fn test_top_k_more_than_available() {
        let mut fv = with_channels(&["ch"]);
        fv.record_activation("ch", 1.0, 0).unwrap();
        let top = fv.top_k_n("ch", 10).unwrap();
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_top_k_empty() {
        let mut fv = with_channels(&["ch"]);
        let top = fv.top_k("ch").unwrap();
        assert!(top.is_empty());
    }

    #[test]
    fn test_top_k_nonexistent() {
        let mut fv = small_fv();
        assert!(fv.top_k("nope").is_err());
    }

    #[test]
    fn test_top_k_stats_incremented() {
        let mut fv = with_channels(&["ch"]);
        fv.record_activation("ch", 1.0, 0).unwrap();
        fv.top_k("ch").unwrap();
        fv.top_k("ch").unwrap();
        assert_eq!(fv.stats().total_top_k_queries, 2);
    }

    // Activation maximization -------------------------------------------------

    #[test]
    fn test_maximize_simple() {
        let mut fv = with_channels(&["ch"]);

        // Activation = sum of squares (maximise by increasing all inputs)
        let activation_fn: ActivationFn =
            Box::new(|input: &[f64]| input.iter().map(|x| x * x).sum::<f64>());

        let result = fv.maximize("ch", &activation_fn, 3).unwrap();
        assert_eq!(result.channel, "ch");
        assert_eq!(result.optimized_input.len(), 3);
        // The optimized input should have moved away from zero
        let norm: f64 = result.optimized_input.iter().map(|x| x * x).sum::<f64>();
        // With L2 decay, it won't go to infinity, but it should be > 0
        // Starting from zero, gradient of sum(x²) at 0 is all zeros,
        // so it might not move. Let's check the final activation.
        // Actually, gradient of sum(x²) w.r.t. x_i = 2*x_i, which is 0 at origin.
        // So this specific function won't move from zero. Let's just verify it ran.
        assert!(result.steps_taken > 0);
        assert!(!result.activation_history.is_empty());
    }

    #[test]
    fn test_maximize_linear() {
        let mut fv = with_channels(&["ch"]);

        // Activation = 3*x[0] + 2*x[1] — gradient is constant and non-zero
        let activation_fn: ActivationFn = Box::new(|input: &[f64]| 3.0 * input[0] + 2.0 * input[1]);

        let result = fv.maximize("ch", &activation_fn, 2).unwrap();
        // After gradient ascent, x[0] should be positive (gradient = 3)
        assert!(
            result.optimized_input[0] > 0.0,
            "x[0] should be positive, got {}",
            result.optimized_input[0]
        );
        // x[1] should also be positive (gradient = 2)
        assert!(
            result.optimized_input[1] > 0.0,
            "x[1] should be positive, got {}",
            result.optimized_input[1]
        );
        // Activation should be positive
        assert!(result.final_activation > 0.0);
    }

    #[test]
    fn test_maximize_from_initial() {
        let mut fv = with_channels(&["ch"]);

        let activation_fn: ActivationFn = Box::new(|input: &[f64]| input[0]);

        let initial = vec![5.0, 0.0];
        let result = fv.maximize_from("ch", &activation_fn, &initial).unwrap();
        // Starting from x[0]=5, with gradient=1, it should increase
        assert!(result.optimized_input[0] > 5.0);
    }

    #[test]
    fn test_maximize_nonexistent_channel() {
        let mut fv = small_fv();
        let f: ActivationFn = Box::new(|_| 0.0);
        assert!(fv.maximize("nope", &f, 3).is_err());
    }

    #[test]
    fn test_maximize_empty_input() {
        let mut fv = with_channels(&["ch"]);
        let f: ActivationFn = Box::new(|_| 0.0);
        assert!(fv.maximize_from("ch", &f, &[]).is_err());
    }

    #[test]
    fn test_maximize_stats_incremented() {
        let mut fv = with_channels(&["ch"]);
        let f: ActivationFn = Box::new(|input: &[f64]| input[0]);
        fv.maximize("ch", &f, 2).unwrap();
        assert_eq!(fv.stats().total_maximizations, 1);
    }

    #[test]
    fn test_maximize_convergence() {
        let mut fv = with_channels(&["ch"]);

        // Constant function — gradient is zero, should converge immediately
        let activation_fn: ActivationFn = Box::new(|_| 42.0);

        let result = fv.maximize("ch", &activation_fn, 2).unwrap();
        assert!(result.converged, "constant function should converge");
        assert!(result.steps_taken <= 3); // should stop very quickly
    }

    #[test]
    fn test_maximize_activation_history() {
        let mut fv = with_channels(&["ch"]);
        let f: ActivationFn = Box::new(|input: &[f64]| input[0]);
        let result = fv.maximize("ch", &f, 1).unwrap();
        // History should be non-empty and values should generally increase
        assert!(!result.activation_history.is_empty());
        if result.activation_history.len() >= 2 {
            let first = result.activation_history[0];
            let last = *result.activation_history.last().unwrap();
            assert!(
                last >= first,
                "activation should increase: first={}, last={}",
                first,
                last
            );
        }
    }

    // Tick & EMA --------------------------------------------------------------

    #[test]
    fn test_tick_increments() {
        let mut fv = small_fv();
        assert_eq!(fv.current_tick(), 0);
        fv.tick();
        assert_eq!(fv.current_tick(), 1);
        fv.tick();
        assert_eq!(fv.current_tick(), 2);
    }

    #[test]
    fn test_ema_initializes_on_first_tick() {
        let mut fv = with_channels(&["ch"]);
        fv.record_activation("ch", 10.0, 0).unwrap();
        fv.tick();
        // EMA should be initialised to the mean importance (10.0 for mean metric)
        assert!((fv.ema_importance() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_ema_smoothing() {
        let mut fv = with_channels(&["ch"]);
        fv.record_activation("ch", 10.0, 0).unwrap();
        fv.tick();
        let ema1 = fv.ema_importance();
        fv.record_activation("ch", 20.0, 1).unwrap();
        fv.tick();
        let ema2 = fv.ema_importance();
        // After recording 20 (mean now 15), EMA should have moved toward 15
        // but not be exactly 15 due to smoothing
        assert!(ema2 > ema1, "EMA should increase: {} -> {}", ema1, ema2);
    }

    #[test]
    fn test_ema_empty_channels() {
        let mut fv = small_fv(); // no channels
        fv.tick();
        assert_eq!(fv.ema_importance(), 0.0);
    }

    // Warmed-up & confidence --------------------------------------------------

    #[test]
    fn test_not_warmed_up_initially() {
        let fv = small_fv();
        assert!(!fv.is_warmed_up());
        assert_eq!(fv.confidence(), 0.0);
    }

    #[test]
    fn test_warmed_up_after_min_samples() {
        let mut fv = small_fv(); // min_samples = 3
        for _ in 0..3 {
            fv.tick();
        }
        assert!(fv.is_warmed_up());
        assert!((fv.confidence() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_confidence_increases() {
        let mut fv = small_fv();
        fv.tick();
        let c1 = fv.confidence();
        fv.tick();
        let c2 = fv.confidence();
        assert!(c2 > c1);
    }

    // Windowed diagnostics ----------------------------------------------------

    #[test]
    fn test_windowed_empty() {
        let fv = small_fv();
        assert_eq!(fv.windowed_mean_importance(), 0.0);
        assert_eq!(fv.windowed_mean_channels_updated(), 0.0);
    }

    #[test]
    fn test_windowed_after_ticks() {
        let mut fv = with_channels(&["ch"]);
        fv.record_activation("ch", 5.0, 0).unwrap();
        fv.tick();
        assert!((fv.windowed_mean_importance() - 5.0).abs() < 1e-10);
        assert!((fv.windowed_mean_channels_updated() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_window_eviction() {
        let mut cfg = small_config();
        cfg.window_size = 3;
        let mut fv = FeatureViz::with_config(cfg).unwrap();
        for _ in 0..5 {
            fv.tick();
        }
        assert_eq!(fv.recent.len(), 3);
    }

    #[test]
    fn test_trend_insufficient_data() {
        let mut fv = small_fv();
        assert!(!fv.is_importance_increasing());
        assert!(!fv.is_importance_decreasing());
        fv.tick();
        assert!(!fv.is_importance_increasing());
    }

    #[test]
    fn test_is_importance_increasing() {
        let mut fv = with_channels(&["ch"]);
        // Record increasing activation values across ticks
        for i in 0..6 {
            fv.record_activation("ch", (i * 10) as f64, i).unwrap();
            fv.tick();
        }
        assert!(fv.is_importance_increasing());
    }

    #[test]
    fn test_is_importance_decreasing() {
        let mut fv = with_channels(&["ch"]);
        for i in 0..6 {
            fv.record_activation("ch", (60 - i * 10) as f64, i).unwrap();
            fv.tick();
        }
        assert!(fv.is_importance_decreasing());
    }

    // Stats -------------------------------------------------------------------

    #[test]
    fn test_stats_initial() {
        let fv = small_fv();
        assert_eq!(fv.stats().total_records, 0);
        assert_eq!(fv.stats().total_maximizations, 0);
        assert_eq!(fv.stats().total_rankings, 0);
        assert_eq!(fv.stats().total_top_k_queries, 0);
        assert_eq!(fv.stats().registered_channels, 0);
        assert_eq!(fv.stats().peak_mean_importance, 0.0);
        assert_eq!(fv.stats().overall_mean_importance(), 0.0);
    }

    #[test]
    fn test_stats_registered_channels() {
        let fv = with_channels(&["a", "b"]);
        assert_eq!(fv.stats().registered_channels, 2);
    }

    #[test]
    fn test_stats_peak_importance() {
        let mut fv = with_channels(&["ch"]);
        fv.record_activation("ch", 100.0, 0).unwrap();
        fv.tick();
        fv.record_activation("ch", 50.0, 1).unwrap();
        fv.tick();
        // Peak should capture the first tick's importance (mean=100),
        // second tick's mean = (100+50)/2 = 75
        assert!(fv.stats().peak_mean_importance >= 75.0);
    }

    #[test]
    fn test_stats_overall_mean() {
        let mut fv = with_channels(&["ch"]);
        fv.record_activation("ch", 10.0, 0).unwrap();
        fv.tick();
        fv.record_activation("ch", 20.0, 1).unwrap();
        fv.tick();
        let overall = fv.stats().overall_mean_importance();
        assert!(overall > 0.0);
    }

    // Reset -------------------------------------------------------------------

    #[test]
    fn test_reset() {
        let mut fv = with_channels(&["a", "b"]);
        fv.record_activation("a", 1.0, 0).unwrap();
        fv.record_activation("b", 2.0, 1).unwrap();
        fv.tick();
        fv.tick();
        fv.reset();
        assert_eq!(fv.channel_count(), 2); // channels preserved
        assert_eq!(fv.current_tick(), 0);
        assert_eq!(fv.stats().total_records, 0);
        assert!(fv.recent.is_empty());
        // Per-channel stats should be reset
        assert_eq!(fv.channel_summary("a").unwrap().count, 0);
    }

    #[test]
    fn test_reset_all() {
        let mut fv = with_channels(&["a", "b"]);
        fv.record_activation("a", 1.0, 0).unwrap();
        fv.tick();
        fv.reset_all();
        assert_eq!(fv.channel_count(), 0);
        assert_eq!(fv.current_tick(), 0);
        assert_eq!(fv.stats().registered_channels, 0);
    }

    // process() compat --------------------------------------------------------

    #[test]
    fn test_process() {
        let fv = small_fv();
        assert!(fv.process().is_ok());
    }

    // Numerical gradient test -------------------------------------------------

    #[test]
    fn test_numerical_gradient_linear() {
        // f(x) = 3*x[0] + 2*x[1]; grad = [3, 2]
        let f: ActivationFn = Box::new(|input: &[f64]| 3.0 * input[0] + 2.0 * input[1]);
        let grad = numerical_gradient(&f, &[1.0, 1.0], 1e-5);
        assert!((grad[0] - 3.0).abs() < 1e-4);
        assert!((grad[1] - 2.0).abs() < 1e-4);
    }

    #[test]
    fn test_numerical_gradient_quadratic() {
        // f(x) = x[0]^2 + x[1]^2; grad at (3,4) = [6, 8]
        let f: ActivationFn = Box::new(|input: &[f64]| input[0] * input[0] + input[1] * input[1]);
        let grad = numerical_gradient(&f, &[3.0, 4.0], 1e-5);
        assert!((grad[0] - 6.0).abs() < 1e-4);
        assert!((grad[1] - 8.0).abs() < 1e-4);
    }

    // ChannelState internal tests ---------------------------------------------

    #[test]
    fn test_channel_state_variance_single() {
        let mut ch = ChannelState::new("test".into(), 100);
        ch.record(5.0, 0);
        assert_eq!(ch.variance(), 0.0);
    }

    #[test]
    fn test_channel_state_variance_multiple() {
        let mut ch = ChannelState::new("test".into(), 100);
        for v in &[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
            ch.record(*v, 0);
        }
        let expected = 32.0 / 7.0;
        assert!((ch.variance() - expected).abs() < 1e-6);
    }

    #[test]
    fn test_channel_state_range_empty() {
        let ch = ChannelState::new("test".into(), 100);
        assert_eq!(ch.range(), 0.0);
    }

    #[test]
    fn test_channel_state_top_k() {
        let mut ch = ChannelState::new("test".into(), 100);
        ch.record(3.0, 1);
        ch.record(1.0, 2);
        ch.record(5.0, 3);
        ch.record(2.0, 4);
        let top = ch.top_k(2);
        assert_eq!(top.len(), 2);
        assert!((top[0].activation - 5.0).abs() < 1e-10);
        assert_eq!(top[0].sample_id, 3);
        assert!((top[1].activation - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_channel_state_reset() {
        let mut ch = ChannelState::new("test".into(), 100);
        ch.record(10.0, 0);
        ch.record(20.0, 1);
        ch.reset_stats();
        assert_eq!(ch.count, 0);
        assert_eq!(ch.mean, 0.0);
        assert!(ch.records.is_empty());
        assert_eq!(ch.min, f64::INFINITY);
        assert_eq!(ch.max, f64::NEG_INFINITY);
    }

    // Edge cases --------------------------------------------------------------

    #[test]
    fn test_negative_activations() {
        let mut fv = with_channels(&["ch"]);
        fv.record_activation("ch", -5.0, 0).unwrap();
        fv.record_activation("ch", -3.0, 1).unwrap();
        let s = fv.channel_summary("ch").unwrap();
        assert!((s.mean - (-4.0)).abs() < 1e-10);
        assert!((s.min - (-5.0)).abs() < 1e-10);
        assert!((s.max - (-3.0)).abs() < 1e-10);
    }

    #[test]
    fn test_constant_activations() {
        let mut fv = with_channels(&["ch"]);
        for _ in 0..10 {
            fv.record_activation("ch", 42.0, 0).unwrap();
        }
        let s = fv.channel_summary("ch").unwrap();
        assert!((s.mean - 42.0).abs() < 1e-10);
        assert!(s.std_dev < 1e-10);
        assert!((s.range).abs() < 1e-10);
    }

    #[test]
    fn test_large_values() {
        let mut fv = with_channels(&["ch"]);
        fv.record_activation("ch", 1e15, 0).unwrap();
        fv.record_activation("ch", -1e15, 1).unwrap();
        let s = fv.channel_summary("ch").unwrap();
        assert!((s.mean).abs() < 1e5); // mean of symmetric values ≈ 0
        assert!((s.range - 2e15).abs() < 1e5);
    }

    // Multiple channels interaction -------------------------------------------

    #[test]
    fn test_independent_channels() {
        let mut fv = with_channels(&["a", "b"]);
        fv.record_activation("a", 100.0, 0).unwrap();
        fv.record_activation("b", 1.0, 0).unwrap();
        let sa = fv.channel_summary("a").unwrap();
        let sb = fv.channel_summary("b").unwrap();
        assert!((sa.mean - 100.0).abs() < 1e-10);
        assert!((sb.mean - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_tick_with_multiple_channels() {
        let mut fv = with_channels(&["a", "b", "c"]);
        fv.record_activation("a", 10.0, 0).unwrap();
        fv.record_activation("b", 20.0, 1).unwrap();
        // "c" has no records — should be excluded from mean importance
        fv.tick();
        // Mean importance = (10 + 20) / 2 = 15
        assert!((fv.ema_importance() - 15.0).abs() < 1e-10);
    }

    // set_importance_metric ---------------------------------------------------

    #[test]
    fn test_set_importance_metric() {
        let mut fv = small_fv();
        assert_eq!(fv.importance_metric(), ImportanceMetric::MeanActivation);
        fv.set_importance_metric(ImportanceMetric::Variance);
        assert_eq!(fv.importance_metric(), ImportanceMetric::Variance);
    }

    // Maximization with custom initial input ----------------------------------

    #[test]
    fn test_maximize_preserves_dimensionality() {
        let mut fv = with_channels(&["ch"]);
        let f: ActivationFn = Box::new(|input: &[f64]| input.iter().sum::<f64>());
        let result = fv.maximize("ch", &f, 5).unwrap();
        assert_eq!(result.optimized_input.len(), 5);
    }

    #[test]
    fn test_maximize_negative_gradient_scenario() {
        let mut fv = with_channels(&["ch"]);

        // Activation = -x[0]^2, peak at x=0, gradient ascent from x=5 should decrease x
        let activation_fn: ActivationFn = Box::new(|input: &[f64]| -(input[0] * input[0]));

        let result = fv.maximize_from("ch", &activation_fn, &[5.0]).unwrap();
        // x should have moved toward 0
        assert!(
            result.optimized_input[0].abs() < 5.0,
            "should move toward 0, got {}",
            result.optimized_input[0]
        );
    }
}

