//! Data buffering and windowing
//!
//! Part of the Visual Cortex region
//! Component: eyes
//!
//! Provides ring-buffer–based data buffering with windowing for incoming
//! market data streams. Supports multiple named channels, configurable
//! window sizes, EMA-smoothed throughput tracking, gap detection, and
//! snapshot extraction for downstream GAF encoding.
//!
//! ## Features
//!
//! - **Ring buffer per channel**: Fixed-capacity circular buffers that
//!   silently evict the oldest sample on overflow
//! - **Windowed extraction**: Retrieve the most recent N samples from
//!   any channel as a contiguous slice
//! - **Multi-channel support**: Register independent channels (e.g.
//!   "price", "volume", "spread") and push data independently
//! - **EMA throughput tracking**: Exponentially weighted moving average
//!   of push rate per tick for backpressure signalling
//! - **Gap detection**: Track sequence numbers per channel and flag
//!   gaps when consecutive pushes skip sequence IDs
//! - **Running statistics**: Per-channel min/max/mean and global stats
//! - **Snapshot API**: Extract aligned windows across multiple channels
//!   for use as GAF encoder input

use crate::common::{Error, Result};
use std::collections::{HashMap, VecDeque};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the buffering engine.
#[derive(Debug, Clone)]
pub struct BufferingConfig {
    /// Maximum samples per channel ring buffer.
    pub max_buffer_size: usize,
    /// Default window size for extraction.
    pub default_window: usize,
    /// Maximum number of channels that can be registered.
    pub max_channels: usize,
    /// EMA decay factor for throughput tracking (0 < decay < 1).
    pub ema_decay: f64,
    /// Minimum samples before EMA is considered initialised.
    pub min_samples: usize,
}

impl Default for BufferingConfig {
    fn default() -> Self {
        Self {
            max_buffer_size: 4096,
            default_window: 64,
            max_channels: 64,
            ema_decay: 0.05,
            min_samples: 10,
        }
    }
}

// ---------------------------------------------------------------------------
// Sample
// ---------------------------------------------------------------------------

/// A single buffered sample.
#[derive(Debug, Clone)]
pub struct Sample {
    /// The numeric value.
    pub value: f64,
    /// Monotonically increasing sequence number (caller-assigned).
    pub sequence: u64,
    /// Tick at which the sample was pushed.
    pub tick: u64,
}

// ---------------------------------------------------------------------------
// Channel
// ---------------------------------------------------------------------------

/// Per-channel ring buffer and statistics.
#[derive(Debug, Clone)]
struct Channel {
    /// Ring buffer of samples.
    buf: VecDeque<Sample>,
    /// Capacity (mirrors config.max_buffer_size for this channel).
    capacity: usize,
    /// Last sequence number seen (for gap detection).
    last_sequence: Option<u64>,
    /// Number of detected gaps.
    gaps_detected: u64,
    /// Total samples pushed (including evicted).
    total_pushed: u64,
    /// Total samples evicted due to overflow.
    total_evicted: u64,
    /// Running sum for online mean.
    sum: f64,
    /// Running min.
    min_value: f64,
    /// Running max.
    max_value: f64,
}

impl Channel {
    fn new(capacity: usize) -> Self {
        Self {
            buf: VecDeque::with_capacity(capacity.min(1024)),
            capacity,
            last_sequence: None,
            gaps_detected: 0,
            total_pushed: 0,
            total_evicted: 0,
            sum: 0.0,
            min_value: f64::INFINITY,
            max_value: f64::NEG_INFINITY,
        }
    }

    fn push(&mut self, value: f64, sequence: u64, tick: u64) -> bool {
        // Gap detection
        let gap = if let Some(last) = self.last_sequence {
            if sequence > last + 1 {
                self.gaps_detected += 1;
                true
            } else {
                false
            }
        } else {
            false
        };
        self.last_sequence = Some(sequence);

        // Evict oldest if at capacity
        if self.buf.len() >= self.capacity {
            if let Some(evicted) = self.buf.pop_front() {
                self.sum -= evicted.value;
                self.total_evicted += 1;
            }
        }

        self.buf.push_back(Sample {
            value,
            sequence,
            tick,
        });
        self.sum += value;
        self.total_pushed += 1;

        if value < self.min_value {
            self.min_value = value;
        }
        if value > self.max_value {
            self.max_value = value;
        }

        gap
    }

    fn len(&self) -> usize {
        self.buf.len()
    }

    fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    fn mean(&self) -> f64 {
        if self.buf.is_empty() {
            0.0
        } else {
            self.sum / self.buf.len() as f64
        }
    }

    fn window(&self, size: usize) -> Vec<f64> {
        let n = self.buf.len();
        let take = size.min(n);
        self.buf
            .iter()
            .skip(n.saturating_sub(take))
            .map(|s| s.value)
            .collect()
    }

    fn window_samples(&self, size: usize) -> Vec<&Sample> {
        let n = self.buf.len();
        let take = size.min(n);
        self.buf.iter().skip(n.saturating_sub(take)).collect()
    }

    fn latest(&self) -> Option<&Sample> {
        self.buf.back()
    }

    fn clear(&mut self) {
        self.buf.clear();
        self.sum = 0.0;
        // Keep statistics across clears
    }

    fn reset(&mut self) {
        self.buf.clear();
        self.sum = 0.0;
        self.last_sequence = None;
        self.gaps_detected = 0;
        self.total_pushed = 0;
        self.total_evicted = 0;
        self.min_value = f64::INFINITY;
        self.max_value = f64::NEG_INFINITY;
    }
}

// ---------------------------------------------------------------------------
// Snapshot
// ---------------------------------------------------------------------------

/// A multi-channel aligned snapshot for downstream consumers.
#[derive(Debug, Clone)]
pub struct Snapshot {
    /// Channel name → window of values.
    pub channels: HashMap<String, Vec<f64>>,
    /// Effective window size (shortest channel length in the snapshot).
    pub window_size: usize,
    /// Tick at which the snapshot was taken.
    pub tick: u64,
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Global buffering statistics.
#[derive(Debug, Clone)]
pub struct BufferingStats {
    /// Total samples pushed across all channels.
    pub total_pushed: u64,
    /// Total samples evicted across all channels.
    pub total_evicted: u64,
    /// Total gaps detected across all channels.
    pub total_gaps: u64,
    /// Number of registered channels.
    pub channel_count: usize,
    /// Total samples currently buffered across all channels.
    pub total_buffered: usize,
    /// EMA of push rate (samples per tick).
    pub ema_throughput: f64,
    /// Number of snapshots taken.
    pub snapshots_taken: u64,
    /// Number of ticks processed.
    pub ticks: u64,
}

impl Default for BufferingStats {
    fn default() -> Self {
        Self {
            total_pushed: 0,
            total_evicted: 0,
            total_gaps: 0,
            channel_count: 0,
            total_buffered: 0,
            ema_throughput: 0.0,
            snapshots_taken: 0,
            ticks: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// ChannelStats (per-channel query result)
// ---------------------------------------------------------------------------

/// Statistics for a single channel.
#[derive(Debug, Clone)]
pub struct ChannelStats {
    /// Number of samples currently buffered.
    pub buffered: usize,
    /// Buffer capacity.
    pub capacity: usize,
    /// Total samples pushed.
    pub total_pushed: u64,
    /// Total samples evicted.
    pub total_evicted: u64,
    /// Number of detected gaps.
    pub gaps_detected: u64,
    /// Current mean of buffered values.
    pub mean: f64,
    /// All-time minimum value.
    pub min_value: f64,
    /// All-time maximum value.
    pub max_value: f64,
    /// Fill ratio (buffered / capacity).
    pub fill_ratio: f64,
}

// ---------------------------------------------------------------------------
// Buffering
// ---------------------------------------------------------------------------

/// Data buffering and windowing engine.
///
/// Ring-buffer–based multi-channel buffer with windowed extraction,
/// EMA throughput tracking, gap detection, and snapshot support.
pub struct Buffering {
    config: BufferingConfig,
    channels: HashMap<String, Channel>,
    channel_order: Vec<String>,

    // EMA throughput state
    ema_throughput: f64,
    ema_initialized: bool,
    pushes_this_tick: u64,

    current_tick: u64,
    snapshots_taken: u64,
}

impl Default for Buffering {
    fn default() -> Self {
        Self::new()
    }
}

impl Buffering {
    /// Create a new buffering engine with default configuration.
    pub fn new() -> Self {
        Self::with_config(BufferingConfig::default()).unwrap()
    }

    /// Create with explicit configuration.
    pub fn with_config(config: BufferingConfig) -> Result<Self> {
        if config.max_buffer_size == 0 {
            return Err(Error::InvalidInput(
                "max_buffer_size must be > 0".into(),
            ));
        }
        if config.default_window == 0 {
            return Err(Error::InvalidInput(
                "default_window must be > 0".into(),
            ));
        }
        if config.max_channels == 0 {
            return Err(Error::InvalidInput(
                "max_channels must be > 0".into(),
            ));
        }
        if config.ema_decay <= 0.0 || config.ema_decay >= 1.0 {
            return Err(Error::InvalidInput(
                "ema_decay must be in (0, 1)".into(),
            ));
        }
        Ok(Self {
            config,
            channels: HashMap::new(),
            channel_order: Vec::new(),
            ema_throughput: 0.0,
            ema_initialized: false,
            pushes_this_tick: 0,
            current_tick: 0,
            snapshots_taken: 0,
        })
    }

    // -----------------------------------------------------------------------
    // Channel management
    // -----------------------------------------------------------------------

    /// Register a new named channel.
    pub fn register_channel(&mut self, name: impl Into<String>) -> Result<()> {
        let name = name.into();
        if self.channels.contains_key(&name) {
            return Err(Error::InvalidInput(format!(
                "channel '{}' already registered",
                name
            )));
        }
        if self.channels.len() >= self.config.max_channels {
            return Err(Error::ResourceExhausted(format!(
                "maximum channels ({}) reached",
                self.config.max_channels
            )));
        }
        self.channels
            .insert(name.clone(), Channel::new(self.config.max_buffer_size));
        self.channel_order.push(name);
        Ok(())
    }

    /// Deregister a channel and discard its data.
    pub fn deregister_channel(&mut self, name: &str) -> Result<()> {
        if self.channels.remove(name).is_none() {
            return Err(Error::NotFound(format!("channel '{}' not found", name)));
        }
        self.channel_order.retain(|n| n != name);
        Ok(())
    }

    /// Get the list of registered channel names (in registration order).
    pub fn channel_names(&self) -> &[String] {
        &self.channel_order
    }

    /// Number of registered channels.
    pub fn channel_count(&self) -> usize {
        self.channels.len()
    }

    // -----------------------------------------------------------------------
    // Push data
    // -----------------------------------------------------------------------

    /// Push a value into a named channel.
    ///
    /// Returns `true` if a gap was detected on this push.
    pub fn push(&mut self, channel: &str, value: f64, sequence: u64) -> Result<bool> {
        let ch = self
            .channels
            .get_mut(channel)
            .ok_or_else(|| Error::NotFound(format!("channel '{}' not found", channel)))?;
        let gap = ch.push(value, sequence, self.current_tick);
        self.pushes_this_tick += 1;
        Ok(gap)
    }

    /// Push a value into a channel, auto-incrementing the sequence number.
    pub fn push_auto(&mut self, channel: &str, value: f64) -> Result<bool> {
        let next_seq = self
            .channels
            .get(channel)
            .ok_or_else(|| Error::NotFound(format!("channel '{}' not found", channel)))?
            .last_sequence
            .map(|s| s + 1)
            .unwrap_or(0);
        self.push(channel, value, next_seq)
    }

    /// Push values into multiple channels simultaneously.
    ///
    /// `entries` is a slice of (channel_name, value, sequence).
    /// Returns the number of gaps detected.
    pub fn push_batch(
        &mut self,
        entries: &[(&str, f64, u64)],
    ) -> Result<usize> {
        let mut gaps = 0;
        for &(channel, value, sequence) in entries {
            if self.push(channel, value, sequence)? {
                gaps += 1;
            }
        }
        Ok(gaps)
    }

    // -----------------------------------------------------------------------
    // Window extraction
    // -----------------------------------------------------------------------

    /// Extract the most recent `window_size` values from a channel.
    ///
    /// If the channel has fewer samples, returns all available.
    pub fn window(&self, channel: &str, window_size: usize) -> Result<Vec<f64>> {
        let ch = self
            .channels
            .get(channel)
            .ok_or_else(|| Error::NotFound(format!("channel '{}' not found", channel)))?;
        Ok(ch.window(window_size))
    }

    /// Extract the most recent values using the default window size.
    pub fn window_default(&self, channel: &str) -> Result<Vec<f64>> {
        self.window(channel, self.config.default_window)
    }

    /// Extract the most recent `window_size` full samples from a channel.
    pub fn window_samples(&self, channel: &str, window_size: usize) -> Result<Vec<&Sample>> {
        let ch = self
            .channels
            .get(channel)
            .ok_or_else(|| Error::NotFound(format!("channel '{}' not found", channel)))?;
        Ok(ch.window_samples(window_size))
    }

    /// Get the latest sample from a channel.
    pub fn latest(&self, channel: &str) -> Result<Option<&Sample>> {
        let ch = self
            .channels
            .get(channel)
            .ok_or_else(|| Error::NotFound(format!("channel '{}' not found", channel)))?;
        Ok(ch.latest())
    }

    // -----------------------------------------------------------------------
    // Snapshot
    // -----------------------------------------------------------------------

    /// Take an aligned snapshot across the given channels.
    ///
    /// The effective window is the minimum of `window_size` and the shortest
    /// channel length among the requested channels.
    pub fn snapshot(
        &mut self,
        channel_names: &[&str],
        window_size: usize,
    ) -> Result<Snapshot> {
        if channel_names.is_empty() {
            return Err(Error::InvalidInput(
                "snapshot requires at least one channel".into(),
            ));
        }

        // Determine effective window size
        let mut effective = window_size;
        for &name in channel_names {
            let ch = self
                .channels
                .get(name)
                .ok_or_else(|| Error::NotFound(format!("channel '{}' not found", name)))?;
            effective = effective.min(ch.len());
        }

        let mut channels = HashMap::new();
        for &name in channel_names {
            let ch = self.channels.get(name).unwrap();
            channels.insert(name.to_string(), ch.window(effective));
        }

        self.snapshots_taken += 1;
        Ok(Snapshot {
            channels,
            window_size: effective,
            tick: self.current_tick,
        })
    }

    /// Take a snapshot using default window and all registered channels.
    pub fn snapshot_all(&mut self) -> Result<Snapshot> {
        let names: Vec<String> = self.channel_order.clone();
        let refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        if refs.is_empty() {
            return Err(Error::InvalidInput(
                "no channels registered for snapshot".into(),
            ));
        }
        self.snapshot(&refs, self.config.default_window)
    }

    // -----------------------------------------------------------------------
    // Queries
    // -----------------------------------------------------------------------

    /// Number of samples currently buffered in a channel.
    pub fn len(&self, channel: &str) -> Result<usize> {
        let ch = self
            .channels
            .get(channel)
            .ok_or_else(|| Error::NotFound(format!("channel '{}' not found", channel)))?;
        Ok(ch.len())
    }

    /// Whether a channel buffer is empty.
    pub fn is_empty(&self, channel: &str) -> Result<bool> {
        let ch = self
            .channels
            .get(channel)
            .ok_or_else(|| Error::NotFound(format!("channel '{}' not found", channel)))?;
        Ok(ch.is_empty())
    }

    /// Fill ratio for a channel (0.0 – 1.0).
    pub fn fill_ratio(&self, channel: &str) -> Result<f64> {
        let ch = self
            .channels
            .get(channel)
            .ok_or_else(|| Error::NotFound(format!("channel '{}' not found", channel)))?;
        Ok(ch.len() as f64 / ch.capacity as f64)
    }

    /// Total samples currently buffered across all channels.
    pub fn total_buffered(&self) -> usize {
        self.channels.values().map(|ch| ch.len()).sum()
    }

    /// Per-channel statistics.
    pub fn channel_stats(&self, channel: &str) -> Result<ChannelStats> {
        let ch = self
            .channels
            .get(channel)
            .ok_or_else(|| Error::NotFound(format!("channel '{}' not found", channel)))?;
        Ok(ChannelStats {
            buffered: ch.len(),
            capacity: ch.capacity,
            total_pushed: ch.total_pushed,
            total_evicted: ch.total_evicted,
            gaps_detected: ch.gaps_detected,
            mean: ch.mean(),
            min_value: ch.min_value,
            max_value: ch.max_value,
            fill_ratio: ch.len() as f64 / ch.capacity as f64,
        })
    }

    /// Whether the EMA throughput tracker has been initialised.
    pub fn is_warmed_up(&self) -> bool {
        self.current_tick >= self.config.min_samples as u64
    }

    /// Current EMA throughput (samples pushed per tick, smoothed).
    pub fn ema_throughput(&self) -> f64 {
        self.ema_throughput
    }

    /// Current tick.
    pub fn current_tick(&self) -> u64 {
        self.current_tick
    }

    // -----------------------------------------------------------------------
    // Tick / Process
    // -----------------------------------------------------------------------

    /// Advance one tick.
    ///
    /// Updates the EMA throughput and resets the per-tick push counter.
    pub fn tick(&mut self) {
        let rate = self.pushes_this_tick as f64;
        if !self.ema_initialized {
            self.ema_throughput = rate;
            self.ema_initialized = true;
        } else {
            self.ema_throughput =
                self.config.ema_decay * rate + (1.0 - self.config.ema_decay) * self.ema_throughput;
        }
        self.pushes_this_tick = 0;
        self.current_tick += 1;
    }

    /// Aggregate global statistics.
    pub fn stats(&self) -> BufferingStats {
        let mut total_pushed = 0u64;
        let mut total_evicted = 0u64;
        let mut total_gaps = 0u64;
        let mut total_buffered = 0usize;

        for ch in self.channels.values() {
            total_pushed += ch.total_pushed;
            total_evicted += ch.total_evicted;
            total_gaps += ch.gaps_detected;
            total_buffered += ch.len();
        }

        BufferingStats {
            total_pushed,
            total_evicted,
            total_gaps,
            channel_count: self.channels.len(),
            total_buffered,
            ema_throughput: self.ema_throughput,
            snapshots_taken: self.snapshots_taken,
            ticks: self.current_tick,
        }
    }

    // -----------------------------------------------------------------------
    // Clear / Reset
    // -----------------------------------------------------------------------

    /// Clear all channel buffers but keep registrations and stats.
    pub fn clear_buffers(&mut self) {
        for ch in self.channels.values_mut() {
            ch.clear();
        }
    }

    /// Clear a single channel's buffer.
    pub fn clear_channel(&mut self, channel: &str) -> Result<()> {
        let ch = self
            .channels
            .get_mut(channel)
            .ok_or_else(|| Error::NotFound(format!("channel '{}' not found", channel)))?;
        ch.clear();
        Ok(())
    }

    /// Full reset — clears all data, stats, and channel registrations.
    pub fn reset(&mut self) {
        self.channels.clear();
        self.channel_order.clear();
        self.ema_throughput = 0.0;
        self.ema_initialized = false;
        self.pushes_this_tick = 0;
        self.current_tick = 0;
        self.snapshots_taken = 0;
    }

    /// Reset statistics only (keep buffered data and registrations).
    pub fn reset_stats(&mut self) {
        for ch in self.channels.values_mut() {
            ch.total_pushed = 0;
            ch.total_evicted = 0;
            ch.gaps_detected = 0;
            ch.min_value = f64::INFINITY;
            ch.max_value = f64::NEG_INFINITY;
        }
        self.ema_throughput = 0.0;
        self.ema_initialized = false;
        self.snapshots_taken = 0;
    }

    /// Main processing function (tick alias).
    pub fn process(&mut self) -> Result<()> {
        self.tick();
        Ok(())
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_basic() {
        let b = Buffering::new();
        assert_eq!(b.channel_count(), 0);
        assert_eq!(b.current_tick(), 0);
    }

    #[test]
    fn test_default() {
        let b = Buffering::default();
        assert_eq!(b.channel_count(), 0);
    }

    #[test]
    fn test_with_config() {
        let cfg = BufferingConfig {
            max_buffer_size: 128,
            default_window: 16,
            max_channels: 4,
            ema_decay: 0.1,
            min_samples: 5,
        };
        let b = Buffering::with_config(cfg).unwrap();
        assert_eq!(b.channel_count(), 0);
    }

    #[test]
    fn test_invalid_config_buffer_size() {
        let mut cfg = BufferingConfig::default();
        cfg.max_buffer_size = 0;
        assert!(Buffering::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_default_window() {
        let mut cfg = BufferingConfig::default();
        cfg.default_window = 0;
        assert!(Buffering::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_max_channels() {
        let mut cfg = BufferingConfig::default();
        cfg.max_channels = 0;
        assert!(Buffering::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_ema_decay_zero() {
        let mut cfg = BufferingConfig::default();
        cfg.ema_decay = 0.0;
        assert!(Buffering::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_ema_decay_one() {
        let mut cfg = BufferingConfig::default();
        cfg.ema_decay = 1.0;
        assert!(Buffering::with_config(cfg).is_err());
    }

    // -----------------------------------------------------------------------
    // Channel management
    // -----------------------------------------------------------------------

    #[test]
    fn test_register_channel() {
        let mut b = Buffering::new();
        b.register_channel("price").unwrap();
        assert_eq!(b.channel_count(), 1);
        assert_eq!(b.channel_names(), &["price"]);
    }

    #[test]
    fn test_register_duplicate() {
        let mut b = Buffering::new();
        b.register_channel("price").unwrap();
        assert!(b.register_channel("price").is_err());
    }

    #[test]
    fn test_register_max_capacity() {
        let cfg = BufferingConfig {
            max_channels: 2,
            ..Default::default()
        };
        let mut b = Buffering::with_config(cfg).unwrap();
        b.register_channel("a").unwrap();
        b.register_channel("b").unwrap();
        assert!(b.register_channel("c").is_err());
    }

    #[test]
    fn test_deregister_channel() {
        let mut b = Buffering::new();
        b.register_channel("price").unwrap();
        b.deregister_channel("price").unwrap();
        assert_eq!(b.channel_count(), 0);
    }

    #[test]
    fn test_deregister_nonexistent() {
        let mut b = Buffering::new();
        assert!(b.deregister_channel("nope").is_err());
    }

    // -----------------------------------------------------------------------
    // Push
    // -----------------------------------------------------------------------

    #[test]
    fn test_push_single() {
        let mut b = Buffering::new();
        b.register_channel("price").unwrap();
        let gap = b.push("price", 100.0, 0).unwrap();
        assert!(!gap);
        assert_eq!(b.len("price").unwrap(), 1);
    }

    #[test]
    fn test_push_nonexistent_channel() {
        let mut b = Buffering::new();
        assert!(b.push("nope", 1.0, 0).is_err());
    }

    #[test]
    fn test_push_auto() {
        let mut b = Buffering::new();
        b.register_channel("vol").unwrap();
        b.push_auto("vol", 10.0).unwrap();
        b.push_auto("vol", 20.0).unwrap();
        b.push_auto("vol", 30.0).unwrap();
        assert_eq!(b.len("vol").unwrap(), 3);
        // No gaps expected with auto-increment
        let stats = b.channel_stats("vol").unwrap();
        assert_eq!(stats.gaps_detected, 0);
    }

    #[test]
    fn test_push_auto_nonexistent() {
        let mut b = Buffering::new();
        assert!(b.push_auto("nope", 1.0).is_err());
    }

    #[test]
    fn test_push_batch() {
        let mut b = Buffering::new();
        b.register_channel("price").unwrap();
        b.register_channel("vol").unwrap();
        let gaps = b
            .push_batch(&[("price", 100.0, 0), ("vol", 500.0, 0)])
            .unwrap();
        assert_eq!(gaps, 0);
        assert_eq!(b.len("price").unwrap(), 1);
        assert_eq!(b.len("vol").unwrap(), 1);
    }

    // -----------------------------------------------------------------------
    // Gap detection
    // -----------------------------------------------------------------------

    #[test]
    fn test_gap_detection() {
        let mut b = Buffering::new();
        b.register_channel("price").unwrap();
        b.push("price", 100.0, 0).unwrap();
        b.push("price", 101.0, 1).unwrap();
        let gap = b.push("price", 103.0, 5).unwrap(); // gap: skipped 2,3,4
        assert!(gap);
        let stats = b.channel_stats("price").unwrap();
        assert_eq!(stats.gaps_detected, 1);
    }

    #[test]
    fn test_no_gap_consecutive() {
        let mut b = Buffering::new();
        b.register_channel("price").unwrap();
        for i in 0..10 {
            let gap = b.push("price", i as f64, i).unwrap();
            assert!(!gap);
        }
    }

    // -----------------------------------------------------------------------
    // Ring buffer eviction
    // -----------------------------------------------------------------------

    #[test]
    fn test_eviction() {
        let cfg = BufferingConfig {
            max_buffer_size: 4,
            ..Default::default()
        };
        let mut b = Buffering::with_config(cfg).unwrap();
        b.register_channel("x").unwrap();
        for i in 0..8u64 {
            b.push("x", i as f64, i).unwrap();
        }
        assert_eq!(b.len("x").unwrap(), 4);
        let stats = b.channel_stats("x").unwrap();
        assert_eq!(stats.total_pushed, 8);
        assert_eq!(stats.total_evicted, 4);
        // Should contain the last 4: 4,5,6,7
        let w = b.window("x", 10).unwrap();
        assert_eq!(w, vec![4.0, 5.0, 6.0, 7.0]);
    }

    // -----------------------------------------------------------------------
    // Window extraction
    // -----------------------------------------------------------------------

    #[test]
    fn test_window() {
        let mut b = Buffering::new();
        b.register_channel("price").unwrap();
        for i in 0..10u64 {
            b.push("price", (i + 1) as f64, i).unwrap();
        }
        let w = b.window("price", 3).unwrap();
        assert_eq!(w, vec![8.0, 9.0, 10.0]);
    }

    #[test]
    fn test_window_larger_than_buffer() {
        let mut b = Buffering::new();
        b.register_channel("price").unwrap();
        b.push("price", 1.0, 0).unwrap();
        b.push("price", 2.0, 1).unwrap();
        let w = b.window("price", 100).unwrap();
        assert_eq!(w, vec![1.0, 2.0]);
    }

    #[test]
    fn test_window_default() {
        let cfg = BufferingConfig {
            default_window: 3,
            ..Default::default()
        };
        let mut b = Buffering::with_config(cfg).unwrap();
        b.register_channel("p").unwrap();
        for i in 0..5u64 {
            b.push("p", i as f64, i).unwrap();
        }
        let w = b.window_default("p").unwrap();
        assert_eq!(w, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_window_nonexistent() {
        let b = Buffering::new();
        assert!(b.window("nope", 5).is_err());
    }

    #[test]
    fn test_window_samples() {
        let mut b = Buffering::new();
        b.register_channel("price").unwrap();
        b.push("price", 10.0, 0).unwrap();
        b.push("price", 20.0, 1).unwrap();
        b.push("price", 30.0, 2).unwrap();
        let samples = b.window_samples("price", 2).unwrap();
        assert_eq!(samples.len(), 2);
        assert!((samples[0].value - 20.0).abs() < 1e-10);
        assert!((samples[1].value - 30.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Latest
    // -----------------------------------------------------------------------

    #[test]
    fn test_latest() {
        let mut b = Buffering::new();
        b.register_channel("price").unwrap();
        assert!(b.latest("price").unwrap().is_none());
        b.push("price", 42.0, 0).unwrap();
        let s = b.latest("price").unwrap().unwrap();
        assert!((s.value - 42.0).abs() < 1e-10);
    }

    #[test]
    fn test_latest_nonexistent() {
        let b = Buffering::new();
        assert!(b.latest("nope").is_err());
    }

    // -----------------------------------------------------------------------
    // Snapshot
    // -----------------------------------------------------------------------

    #[test]
    fn test_snapshot() {
        let mut b = Buffering::new();
        b.register_channel("price").unwrap();
        b.register_channel("vol").unwrap();
        for i in 0..5u64 {
            b.push("price", (i + 1) as f64 * 100.0, i).unwrap();
            b.push("vol", (i + 1) as f64 * 1000.0, i).unwrap();
        }
        let snap = b.snapshot(&["price", "vol"], 3).unwrap();
        assert_eq!(snap.window_size, 3);
        assert_eq!(snap.channels["price"], vec![300.0, 400.0, 500.0]);
        assert_eq!(snap.channels["vol"], vec![3000.0, 4000.0, 5000.0]);
    }

    #[test]
    fn test_snapshot_uneven_channels() {
        let mut b = Buffering::new();
        b.register_channel("price").unwrap();
        b.register_channel("vol").unwrap();
        for i in 0..5u64 {
            b.push("price", i as f64, i).unwrap();
        }
        b.push("vol", 1.0, 0).unwrap();
        // Effective window should be min(requested=10, price=5, vol=1) = 1
        let snap = b.snapshot(&["price", "vol"], 10).unwrap();
        assert_eq!(snap.window_size, 1);
        assert_eq!(snap.channels["price"].len(), 1);
        assert_eq!(snap.channels["vol"].len(), 1);
    }

    #[test]
    fn test_snapshot_empty_list() {
        let mut b = Buffering::new();
        assert!(b.snapshot(&[], 5).is_err());
    }

    #[test]
    fn test_snapshot_nonexistent_channel() {
        let mut b = Buffering::new();
        b.register_channel("price").unwrap();
        assert!(b.snapshot(&["price", "nope"], 5).is_err());
    }

    #[test]
    fn test_snapshot_all() {
        let cfg = BufferingConfig {
            default_window: 3,
            ..Default::default()
        };
        let mut b = Buffering::with_config(cfg).unwrap();
        b.register_channel("a").unwrap();
        b.register_channel("b").unwrap();
        for i in 0..5u64 {
            b.push("a", i as f64, i).unwrap();
            b.push("b", (i * 10) as f64, i).unwrap();
        }
        let snap = b.snapshot_all().unwrap();
        assert_eq!(snap.window_size, 3);
        assert_eq!(snap.channels.len(), 2);
    }

    #[test]
    fn test_snapshot_all_empty() {
        let mut b = Buffering::new();
        assert!(b.snapshot_all().is_err());
    }

    #[test]
    fn test_snapshot_count() {
        let mut b = Buffering::new();
        b.register_channel("x").unwrap();
        b.push("x", 1.0, 0).unwrap();
        let _ = b.snapshot(&["x"], 1).unwrap();
        let _ = b.snapshot(&["x"], 1).unwrap();
        assert_eq!(b.stats().snapshots_taken, 2);
    }

    // -----------------------------------------------------------------------
    // Channel stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_channel_stats() {
        let mut b = Buffering::new();
        b.register_channel("price").unwrap();
        b.push("price", 10.0, 0).unwrap();
        b.push("price", 20.0, 1).unwrap();
        b.push("price", 30.0, 2).unwrap();
        let stats = b.channel_stats("price").unwrap();
        assert_eq!(stats.buffered, 3);
        assert_eq!(stats.total_pushed, 3);
        assert_eq!(stats.total_evicted, 0);
        assert!((stats.mean - 20.0).abs() < 1e-10);
        assert!((stats.min_value - 10.0).abs() < 1e-10);
        assert!((stats.max_value - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_channel_stats_nonexistent() {
        let b = Buffering::new();
        assert!(b.channel_stats("nope").is_err());
    }

    #[test]
    fn test_fill_ratio() {
        let cfg = BufferingConfig {
            max_buffer_size: 10,
            ..Default::default()
        };
        let mut b = Buffering::with_config(cfg).unwrap();
        b.register_channel("x").unwrap();
        for i in 0..5u64 {
            b.push("x", i as f64, i).unwrap();
        }
        let ratio = b.fill_ratio("x").unwrap();
        assert!((ratio - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_fill_ratio_nonexistent() {
        let b = Buffering::new();
        assert!(b.fill_ratio("nope").is_err());
    }

    // -----------------------------------------------------------------------
    // Queries
    // -----------------------------------------------------------------------

    #[test]
    fn test_is_empty() {
        let mut b = Buffering::new();
        b.register_channel("x").unwrap();
        assert!(b.is_empty("x").unwrap());
        b.push("x", 1.0, 0).unwrap();
        assert!(!b.is_empty("x").unwrap());
    }

    #[test]
    fn test_is_empty_nonexistent() {
        let b = Buffering::new();
        assert!(b.is_empty("nope").is_err());
    }

    #[test]
    fn test_total_buffered() {
        let mut b = Buffering::new();
        b.register_channel("a").unwrap();
        b.register_channel("b").unwrap();
        b.push("a", 1.0, 0).unwrap();
        b.push("a", 2.0, 1).unwrap();
        b.push("b", 3.0, 0).unwrap();
        assert_eq!(b.total_buffered(), 3);
    }

    // -----------------------------------------------------------------------
    // Tick / EMA throughput
    // -----------------------------------------------------------------------

    #[test]
    fn test_tick_increments() {
        let mut b = Buffering::new();
        assert_eq!(b.current_tick(), 0);
        b.tick();
        assert_eq!(b.current_tick(), 1);
        b.tick();
        assert_eq!(b.current_tick(), 2);
    }

    #[test]
    fn test_ema_throughput_initial() {
        let mut b = Buffering::new();
        b.register_channel("x").unwrap();
        b.push("x", 1.0, 0).unwrap();
        b.push("x", 2.0, 1).unwrap();
        b.push("x", 3.0, 2).unwrap();
        b.tick(); // 3 pushes this tick
        assert!((b.ema_throughput() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_ema_throughput_smoothing() {
        let mut b = Buffering::new();
        b.register_channel("x").unwrap();
        // Tick 1: 10 pushes
        for i in 0..10u64 {
            b.push("x", i as f64, i).unwrap();
        }
        b.tick();
        assert!((b.ema_throughput() - 10.0).abs() < 1e-10);
        // Tick 2: 0 pushes — EMA should decay towards 0
        b.tick();
        assert!(b.ema_throughput() < 10.0);
        assert!(b.ema_throughput() > 0.0);
    }

    #[test]
    fn test_is_warmed_up() {
        let cfg = BufferingConfig {
            min_samples: 3,
            ..Default::default()
        };
        let mut b = Buffering::with_config(cfg).unwrap();
        assert!(!b.is_warmed_up());
        b.tick();
        b.tick();
        assert!(!b.is_warmed_up());
        b.tick();
        assert!(b.is_warmed_up());
    }

    // -----------------------------------------------------------------------
    // Global stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats() {
        let mut b = Buffering::new();
        b.register_channel("a").unwrap();
        b.register_channel("b").unwrap();
        b.push("a", 1.0, 0).unwrap();
        b.push("b", 2.0, 0).unwrap();
        b.push("b", 3.0, 5).unwrap(); // gap
        b.tick();

        let stats = b.stats();
        assert_eq!(stats.total_pushed, 3);
        assert_eq!(stats.total_gaps, 1);
        assert_eq!(stats.channel_count, 2);
        assert_eq!(stats.total_buffered, 3);
        assert_eq!(stats.ticks, 1);
    }

    // -----------------------------------------------------------------------
    // Clear / Reset
    // -----------------------------------------------------------------------

    #[test]
    fn test_clear_buffers() {
        let mut b = Buffering::new();
        b.register_channel("x").unwrap();
        b.push("x", 1.0, 0).unwrap();
        b.clear_buffers();
        assert_eq!(b.len("x").unwrap(), 0);
        // Channel still registered
        assert_eq!(b.channel_count(), 1);
    }

    #[test]
    fn test_clear_channel() {
        let mut b = Buffering::new();
        b.register_channel("a").unwrap();
        b.register_channel("b").unwrap();
        b.push("a", 1.0, 0).unwrap();
        b.push("b", 2.0, 0).unwrap();
        b.clear_channel("a").unwrap();
        assert_eq!(b.len("a").unwrap(), 0);
        assert_eq!(b.len("b").unwrap(), 1);
    }

    #[test]
    fn test_clear_channel_nonexistent() {
        let mut b = Buffering::new();
        assert!(b.clear_channel("nope").is_err());
    }

    #[test]
    fn test_reset() {
        let mut b = Buffering::new();
        b.register_channel("x").unwrap();
        b.push("x", 1.0, 0).unwrap();
        b.tick();
        b.reset();
        assert_eq!(b.channel_count(), 0);
        assert_eq!(b.current_tick(), 0);
        assert!((b.ema_throughput() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_reset_stats() {
        let mut b = Buffering::new();
        b.register_channel("x").unwrap();
        b.push("x", 1.0, 0).unwrap();
        b.tick();
        b.reset_stats();
        let stats = b.stats();
        assert_eq!(stats.total_pushed, 0);
        assert_eq!(stats.snapshots_taken, 0);
        assert!((stats.ema_throughput - 0.0).abs() < 1e-10);
        // Data still present
        assert_eq!(b.len("x").unwrap(), 1);
    }

    // -----------------------------------------------------------------------
    // Process
    // -----------------------------------------------------------------------

    #[test]
    fn test_process() {
        let mut b = Buffering::new();
        assert!(b.process().is_ok());
        assert_eq!(b.current_tick(), 1);
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_mean_after_eviction() {
        let cfg = BufferingConfig {
            max_buffer_size: 3,
            ..Default::default()
        };
        let mut b = Buffering::with_config(cfg).unwrap();
        b.register_channel("x").unwrap();
        // Push 1, 2, 3 (mean = 2.0)
        b.push("x", 1.0, 0).unwrap();
        b.push("x", 2.0, 1).unwrap();
        b.push("x", 3.0, 2).unwrap();
        let m1 = b.channel_stats("x").unwrap().mean;
        assert!((m1 - 2.0).abs() < 1e-10);
        // Push 10 — evicts 1.0, buffer is [2, 3, 10], mean = 5.0
        b.push("x", 10.0, 3).unwrap();
        let m2 = b.channel_stats("x").unwrap().mean;
        assert!((m2 - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_min_max_survive_eviction() {
        let cfg = BufferingConfig {
            max_buffer_size: 2,
            ..Default::default()
        };
        let mut b = Buffering::with_config(cfg).unwrap();
        b.register_channel("x").unwrap();
        b.push("x", 100.0, 0).unwrap();
        b.push("x", 200.0, 1).unwrap();
        b.push("x", 50.0, 2).unwrap(); // evicts 100.0
        let stats = b.channel_stats("x").unwrap();
        // min/max are all-time
        assert!((stats.min_value - 50.0).abs() < 1e-10);
        assert!((stats.max_value - 200.0).abs() < 1e-10);
    }

    #[test]
    fn test_multiple_gaps() {
        let mut b = Buffering::new();
        b.register_channel("x").unwrap();
        b.push("x", 1.0, 0).unwrap();
        b.push("x", 2.0, 5).unwrap(); // gap 1
        b.push("x", 3.0, 6).unwrap(); // no gap
        b.push("x", 4.0, 10).unwrap(); // gap 2
        let stats = b.channel_stats("x").unwrap();
        assert_eq!(stats.gaps_detected, 2);
    }

    #[test]
    fn test_push_batch_with_gap() {
        let mut b = Buffering::new();
        b.register_channel("x").unwrap();
        b.push("x", 1.0, 0).unwrap();
        let gaps = b.push_batch(&[("x", 2.0, 5), ("x", 3.0, 6)]).unwrap();
        assert_eq!(gaps, 1);
    }

    #[test]
    fn test_snapshot_tick_recorded() {
        let mut b = Buffering::new();
        b.register_channel("x").unwrap();
        b.push("x", 1.0, 0).unwrap();
        b.tick();
        b.tick();
        let snap = b.snapshot(&["x"], 1).unwrap();
        assert_eq!(snap.tick, 2);
    }

    #[test]
    fn test_window_empty_channel() {
        let mut b = Buffering::new();
        b.register_channel("x").unwrap();
        let w = b.window("x", 5).unwrap();
        assert!(w.is_empty());
    }

    #[test]
    fn test_channel_order_preserved() {
        let mut b = Buffering::new();
        b.register_channel("c").unwrap();
        b.register_channel("a").unwrap();
        b.register_channel("b").unwrap();
        assert_eq!(b.channel_names(), &["c", "a", "b"]);
    }

    #[test]
    fn test_deregister_preserves_order() {
        let mut b = Buffering::new();
        b.register_channel("a").unwrap();
        b.register_channel("b").unwrap();
        b.register_channel("c").unwrap();
        b.deregister_channel("b").unwrap();
        assert_eq!(b.channel_names(), &["a", "c"]);
    }
}
