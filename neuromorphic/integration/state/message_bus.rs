//! Inter-region pub/sub messaging with channels and topic-based routing.
//!
//! Part of the Integration region — State component.
//!
//! `MessageBus` provides a lightweight, in-process publish/subscribe messaging
//! system for communication between brain regions. Messages are routed by
//! topic, buffered in per-channel queues, and delivered in FIFO order.
//! The module tracks EMA-smoothed throughput metrics and windowed diagnostics.

use crate::common::Result;
use std::collections::{HashMap, VecDeque};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the message bus.
#[derive(Debug, Clone)]
pub struct MessageBusConfig {
    /// Maximum number of channels that can be created.
    pub max_channels: usize,
    /// Maximum number of messages buffered per channel before oldest are dropped.
    pub channel_buffer_size: usize,
    /// Maximum number of topics that can be tracked.
    pub max_topics: usize,
    /// EMA decay factor (0 < α < 1). Higher → faster adaptation.
    pub ema_decay: f64,
    /// Window size for rolling diagnostics.
    pub window_size: usize,
}

impl Default for MessageBusConfig {
    fn default() -> Self {
        Self {
            max_channels: 256,
            channel_buffer_size: 1024,
            max_topics: 128,
            ema_decay: 0.1,
            window_size: 50,
        }
    }
}

// ---------------------------------------------------------------------------
// Message
// ---------------------------------------------------------------------------

/// A message flowing through the bus.
#[derive(Debug, Clone)]
pub struct Message {
    /// Unique, monotonically increasing message identifier.
    pub id: u64,
    /// Topic string (e.g. `"cortex.signal"`, `"risk.alert"`).
    pub topic: String,
    /// Name of the sending region / component.
    pub sender: String,
    /// Numeric payload (domain-specific meaning).
    pub payload: f64,
    /// Optional human-readable body.
    pub body: Option<String>,
    /// Tick at which the message was sent.
    pub tick: u64,
}

// ---------------------------------------------------------------------------
// Channel
// ---------------------------------------------------------------------------

/// A named communication channel with its own message buffer.
#[derive(Debug, Clone)]
pub struct Channel {
    /// Channel name (unique within the bus).
    pub name: String,
    /// Topic filter — messages whose topic starts with this prefix are routed
    /// to this channel. An empty string matches all topics.
    pub topic_filter: String,
    /// Buffered messages awaiting consumption.
    buffer: VecDeque<Message>,
    /// Maximum buffer capacity.
    capacity: usize,
    /// Total messages ever enqueued to this channel.
    pub total_enqueued: u64,
    /// Total messages consumed (drained) from this channel.
    pub total_consumed: u64,
    /// Total messages dropped due to buffer overflow.
    pub total_dropped: u64,
}

impl Channel {
    fn new(name: impl Into<String>, topic_filter: impl Into<String>, capacity: usize) -> Self {
        Self {
            name: name.into(),
            topic_filter: topic_filter.into(),
            buffer: VecDeque::with_capacity(capacity.min(256)),
            capacity,
            total_enqueued: 0,
            total_consumed: 0,
            total_dropped: 0,
        }
    }

    /// Whether this channel accepts a message with the given topic.
    fn accepts(&self, topic: &str) -> bool {
        if self.topic_filter.is_empty() {
            return true;
        }
        topic.starts_with(&self.topic_filter)
    }

    /// Push a message into the buffer, dropping the oldest if at capacity.
    fn enqueue(&mut self, msg: Message) -> bool {
        let dropped = if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
            self.total_dropped += 1;
            true
        } else {
            false
        };
        self.buffer.push_back(msg);
        self.total_enqueued += 1;
        dropped
    }

    /// Drain all buffered messages.
    fn drain(&mut self) -> Vec<Message> {
        let msgs: Vec<Message> = self.buffer.drain(..).collect();
        self.total_consumed += msgs.len() as u64;
        msgs
    }

    /// Peek at the next message without consuming it.
    fn peek(&self) -> Option<&Message> {
        self.buffer.front()
    }

    /// Number of buffered messages.
    fn depth(&self) -> usize {
        self.buffer.len()
    }
}

// ---------------------------------------------------------------------------
// Tick snapshot (windowed diagnostics)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct TickSnapshot {
    messages_sent: u64,
    messages_delivered: u64,
    messages_dropped: u64,
    total_buffer_depth: usize,
    active_channels: usize,
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Cumulative statistics for the message bus.
#[derive(Debug, Clone)]
pub struct MessageBusStats {
    /// Total number of messages sent (published).
    pub total_sent: u64,
    /// Total number of individual channel deliveries.
    pub total_delivered: u64,
    /// Total number of messages dropped (buffer overflow or no matching channel).
    pub total_dropped: u64,
    /// Total number of channels ever created.
    pub total_channels_created: u64,
    /// Total number of channels removed.
    pub total_channels_removed: u64,
    /// Total number of drain operations.
    pub total_drains: u64,
    /// EMA-smoothed messages-per-tick send rate.
    pub ema_send_rate: f64,
    /// EMA-smoothed deliveries-per-tick rate.
    pub ema_delivery_rate: f64,
    /// EMA-smoothed total buffer depth across all channels.
    pub ema_buffer_depth: f64,
}

impl Default for MessageBusStats {
    fn default() -> Self {
        Self {
            total_sent: 0,
            total_delivered: 0,
            total_dropped: 0,
            total_channels_created: 0,
            total_channels_removed: 0,
            total_drains: 0,
            ema_send_rate: 0.0,
            ema_delivery_rate: 0.0,
            ema_buffer_depth: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// MessageBus
// ---------------------------------------------------------------------------

/// Inter-region pub/sub message bus.
///
/// Provides:
/// * Named channels with topic-prefix routing.
/// * Buffered message delivery with configurable per-channel capacity.
/// * Tick-based lifecycle with EMA and windowed diagnostics.
pub struct MessageBus {
    config: MessageBusConfig,
    /// Channel name → Channel.
    channels: HashMap<String, Channel>,
    /// Per-topic send counts.
    topic_counts: HashMap<String, u64>,
    /// Monotonically increasing message id counter.
    next_msg_id: u64,
    /// Current tick.
    tick: u64,
    /// Per-tick counters (reset each tick).
    tick_sent: u64,
    tick_delivered: u64,
    tick_dropped: u64,
    /// EMA initialisation flag.
    ema_initialized: bool,
    /// Rolling window of tick snapshots.
    recent: VecDeque<TickSnapshot>,
    /// Cumulative statistics.
    stats: MessageBusStats,
}

impl Default for MessageBus {
    fn default() -> Self {
        Self::new()
    }
}

impl MessageBus {
    // -------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------

    /// Create with default configuration.
    pub fn new() -> Self {
        Self::with_config(MessageBusConfig::default())
    }

    /// Create with explicit configuration.
    pub fn with_config(config: MessageBusConfig) -> Self {
        Self {
            channels: HashMap::with_capacity(config.max_channels.min(32)),
            topic_counts: HashMap::with_capacity(config.max_topics.min(32)),
            next_msg_id: 1,
            tick: 0,
            tick_sent: 0,
            tick_delivered: 0,
            tick_dropped: 0,
            ema_initialized: false,
            recent: VecDeque::with_capacity(config.window_size),
            stats: MessageBusStats::default(),
            config,
        }
    }

    // -------------------------------------------------------------------
    // Channel management
    // -------------------------------------------------------------------

    /// Create a named channel with a topic filter.
    ///
    /// Returns `true` if the channel was created, `false` if a channel with
    /// that name already exists or if the bus is at capacity.
    pub fn create_channel(
        &mut self,
        name: impl Into<String>,
        topic_filter: impl Into<String>,
    ) -> bool {
        let name = name.into();
        if self.channels.contains_key(&name) {
            return false;
        }
        if self.channels.len() >= self.config.max_channels {
            return false;
        }
        let ch = Channel::new(&name, topic_filter, self.config.channel_buffer_size);
        self.channels.insert(name, ch);
        self.stats.total_channels_created += 1;
        true
    }

    /// Remove a channel by name. Returns `true` if found and removed.
    pub fn remove_channel(&mut self, name: &str) -> bool {
        let removed = self.channels.remove(name).is_some();
        if removed {
            self.stats.total_channels_removed += 1;
        }
        removed
    }

    /// Number of currently active channels.
    pub fn channel_count(&self) -> usize {
        self.channels.len()
    }

    /// Get a reference to a channel by name.
    pub fn channel(&self, name: &str) -> Option<&Channel> {
        self.channels.get(name)
    }

    /// Get the buffer depth of a specific channel.
    pub fn channel_depth(&self, name: &str) -> usize {
        self.channels.get(name).map_or(0, |ch| ch.depth())
    }

    /// Total buffer depth across all channels.
    pub fn total_buffer_depth(&self) -> usize {
        self.channels.values().map(|ch| ch.depth()).sum()
    }

    /// List all channel names.
    pub fn channel_names(&self) -> Vec<String> {
        self.channels.keys().cloned().collect()
    }

    // -------------------------------------------------------------------
    // Sending
    // -------------------------------------------------------------------

    /// Send a message to all matching channels.
    ///
    /// Returns the message id. The message is cloned into each matching
    /// channel's buffer.
    pub fn send(
        &mut self,
        topic: impl Into<String>,
        sender: impl Into<String>,
        payload: f64,
        body: Option<String>,
    ) -> u64 {
        let topic = topic.into();
        let sender = sender.into();
        let id = self.next_msg_id;
        self.next_msg_id += 1;

        // Track topic counts
        if self.topic_counts.len() < self.config.max_topics
            || self.topic_counts.contains_key(&topic)
        {
            *self.topic_counts.entry(topic.clone()).or_insert(0) += 1;
        }

        let msg = Message {
            id,
            topic: topic.clone(),
            sender,
            payload,
            body,
            tick: self.tick,
        };

        let mut matched = false;
        let channel_names: Vec<String> = self.channels.keys().cloned().collect();

        for ch_name in &channel_names {
            if let Some(ch) = self.channels.get_mut(ch_name) {
                if ch.accepts(&topic) {
                    let dropped = ch.enqueue(msg.clone());
                    self.tick_delivered += 1;
                    self.stats.total_delivered += 1;
                    if dropped {
                        self.tick_dropped += 1;
                        self.stats.total_dropped += 1;
                    }
                    matched = true;
                }
            }
        }

        if !matched {
            self.tick_dropped += 1;
            self.stats.total_dropped += 1;
        }

        self.tick_sent += 1;
        self.stats.total_sent += 1;

        id
    }

    /// Convenience: send a message with only topic, sender, and payload.
    pub fn emit(
        &mut self,
        topic: impl Into<String>,
        sender: impl Into<String>,
        payload: f64,
    ) -> u64 {
        self.send(topic, sender, payload, None)
    }

    /// Number of distinct topics seen.
    pub fn topic_count(&self) -> usize {
        self.topic_counts.len()
    }

    /// Get the send count for a specific topic.
    pub fn topic_send_count(&self, topic: &str) -> u64 {
        self.topic_counts.get(topic).copied().unwrap_or(0)
    }

    // -------------------------------------------------------------------
    // Receiving
    // -------------------------------------------------------------------

    /// Drain all buffered messages from a channel.
    ///
    /// Returns an empty vec if the channel doesn't exist or has no messages.
    pub fn drain(&mut self, channel_name: &str) -> Vec<Message> {
        self.stats.total_drains += 1;
        if let Some(ch) = self.channels.get_mut(channel_name) {
            ch.drain()
        } else {
            Vec::new()
        }
    }

    /// Peek at the next message in a channel without consuming it.
    pub fn peek(&self, channel_name: &str) -> Option<&Message> {
        self.channels.get(channel_name).and_then(|ch| ch.peek())
    }

    /// Drain all messages from all channels, returning them grouped by channel.
    pub fn drain_all(&mut self) -> HashMap<String, Vec<Message>> {
        self.stats.total_drains += 1;
        let names: Vec<String> = self.channels.keys().cloned().collect();
        let mut result = HashMap::new();
        for name in names {
            if let Some(ch) = self.channels.get_mut(&name) {
                let msgs = ch.drain();
                if !msgs.is_empty() {
                    result.insert(name, msgs);
                }
            }
        }
        result
    }

    // -------------------------------------------------------------------
    // Tick lifecycle
    // -------------------------------------------------------------------

    /// Advance the bus tick and update diagnostics.
    ///
    /// Call once per system tick after sending / receiving.
    pub fn tick(&mut self) {
        self.tick += 1;

        let total_depth = self.total_buffer_depth();
        let active = self.channels.values().filter(|ch| ch.depth() > 0).count();

        let snapshot = TickSnapshot {
            messages_sent: self.tick_sent,
            messages_delivered: self.tick_delivered,
            messages_dropped: self.tick_dropped,
            total_buffer_depth: total_depth,
            active_channels: active,
        };

        // EMA update
        let alpha = self.config.ema_decay;
        if !self.ema_initialized {
            self.stats.ema_send_rate = snapshot.messages_sent as f64;
            self.stats.ema_delivery_rate = snapshot.messages_delivered as f64;
            self.stats.ema_buffer_depth = snapshot.total_buffer_depth as f64;
            self.ema_initialized = true;
        } else {
            self.stats.ema_send_rate =
                alpha * snapshot.messages_sent as f64 + (1.0 - alpha) * self.stats.ema_send_rate;
            self.stats.ema_delivery_rate = alpha * snapshot.messages_delivered as f64
                + (1.0 - alpha) * self.stats.ema_delivery_rate;
            self.stats.ema_buffer_depth = alpha * snapshot.total_buffer_depth as f64
                + (1.0 - alpha) * self.stats.ema_buffer_depth;
        }

        // Windowed
        if self.recent.len() >= self.config.window_size {
            self.recent.pop_front();
        }
        self.recent.push_back(snapshot);

        // Reset per-tick counters
        self.tick_sent = 0;
        self.tick_delivered = 0;
        self.tick_dropped = 0;
    }

    /// Current tick value.
    pub fn current_tick(&self) -> u64 {
        self.tick
    }

    /// Main processing function: advance tick.
    pub fn process(&mut self) -> Result<()> {
        self.tick();
        Ok(())
    }

    // -------------------------------------------------------------------
    // Statistics & diagnostics
    // -------------------------------------------------------------------

    /// Reference to cumulative statistics.
    pub fn stats(&self) -> &MessageBusStats {
        &self.stats
    }

    /// Reference to configuration.
    pub fn config(&self) -> &MessageBusConfig {
        &self.config
    }

    /// EMA-smoothed send rate (messages per tick).
    pub fn smoothed_send_rate(&self) -> f64 {
        self.stats.ema_send_rate
    }

    /// EMA-smoothed delivery rate (channel deliveries per tick).
    pub fn smoothed_delivery_rate(&self) -> f64 {
        self.stats.ema_delivery_rate
    }

    /// EMA-smoothed total buffer depth.
    pub fn smoothed_buffer_depth(&self) -> f64 {
        self.stats.ema_buffer_depth
    }

    /// Windowed average send rate.
    pub fn windowed_send_rate(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.recent.iter().map(|s| s.messages_sent as f64).sum();
        sum / self.recent.len() as f64
    }

    /// Windowed average delivery rate.
    pub fn windowed_delivery_rate(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let sum: f64 = self
            .recent
            .iter()
            .map(|s| s.messages_delivered as f64)
            .sum();
        sum / self.recent.len() as f64
    }

    /// Windowed average buffer depth.
    pub fn windowed_buffer_depth(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let sum: f64 = self
            .recent
            .iter()
            .map(|s| s.total_buffer_depth as f64)
            .sum();
        sum / self.recent.len() as f64
    }

    /// Windowed drop rate (fraction of sent messages that were dropped).
    pub fn windowed_drop_rate(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let total_sent: u64 = self.recent.iter().map(|s| s.messages_sent).sum();
        let total_dropped: u64 = self.recent.iter().map(|s| s.messages_dropped).sum();
        if total_sent == 0 {
            return 0.0;
        }
        total_dropped as f64 / total_sent as f64
    }

    /// Whether the send rate is trending upward over the window.
    pub fn is_send_rate_increasing(&self) -> bool {
        if self.recent.len() < 4 {
            return false;
        }
        let n = self.recent.len();
        let half = n / 2;
        let first_avg: f64 = self
            .recent
            .iter()
            .take(half)
            .map(|s| s.messages_sent as f64)
            .sum::<f64>()
            / half as f64;
        let second_avg: f64 = self
            .recent
            .iter()
            .skip(half)
            .map(|s| s.messages_sent as f64)
            .sum::<f64>()
            / (n - half) as f64;
        second_avg > first_avg
    }

    /// Whether buffer depth is trending upward (potential backpressure).
    pub fn is_buffer_growing(&self) -> bool {
        if self.recent.len() < 4 {
            return false;
        }
        let n = self.recent.len();
        let half = n / 2;
        let first_avg: f64 = self
            .recent
            .iter()
            .take(half)
            .map(|s| s.total_buffer_depth as f64)
            .sum::<f64>()
            / half as f64;
        let second_avg: f64 = self
            .recent
            .iter()
            .skip(half)
            .map(|s| s.total_buffer_depth as f64)
            .sum::<f64>()
            / (n - half) as f64;
        second_avg > first_avg
    }

    /// Reset all state (channels, stats, ticks).
    pub fn reset(&mut self) {
        self.channels.clear();
        self.topic_counts.clear();
        self.next_msg_id = 1;
        self.tick = 0;
        self.tick_sent = 0;
        self.tick_delivered = 0;
        self.tick_dropped = 0;
        self.ema_initialized = false;
        self.recent.clear();
        self.stats = MessageBusStats::default();
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> MessageBusConfig {
        MessageBusConfig {
            max_channels: 4,
            channel_buffer_size: 16,
            max_topics: 8,
            ema_decay: 0.5,
            window_size: 5,
        }
    }

    // -------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------

    #[test]
    fn test_new_default() {
        let bus = MessageBus::new();
        assert_eq!(bus.channel_count(), 0);
        assert_eq!(bus.current_tick(), 0);
        assert_eq!(bus.total_buffer_depth(), 0);
    }

    #[test]
    fn test_with_config() {
        let bus = MessageBus::with_config(small_config());
        assert_eq!(bus.config().max_channels, 4);
        assert_eq!(bus.config().channel_buffer_size, 16);
    }

    // -------------------------------------------------------------------
    // Channel management
    // -------------------------------------------------------------------

    #[test]
    fn test_create_channel() {
        let mut bus = MessageBus::with_config(small_config());
        assert!(bus.create_channel("risk", "risk."));
        assert_eq!(bus.channel_count(), 1);
        assert_eq!(bus.stats().total_channels_created, 1);
    }

    #[test]
    fn test_create_duplicate_channel() {
        let mut bus = MessageBus::with_config(small_config());
        assert!(bus.create_channel("risk", "risk."));
        assert!(!bus.create_channel("risk", "risk."));
        assert_eq!(bus.channel_count(), 1);
    }

    #[test]
    fn test_create_channel_at_capacity() {
        let mut bus = MessageBus::with_config(small_config()); // max = 4
        for i in 0..4 {
            assert!(bus.create_channel(format!("ch{}", i), ""));
        }
        assert!(!bus.create_channel("overflow", ""));
        assert_eq!(bus.channel_count(), 4);
    }

    #[test]
    fn test_remove_channel() {
        let mut bus = MessageBus::with_config(small_config());
        bus.create_channel("risk", "risk.");
        assert!(bus.remove_channel("risk"));
        assert_eq!(bus.channel_count(), 0);
        assert_eq!(bus.stats().total_channels_removed, 1);
    }

    #[test]
    fn test_remove_nonexistent_channel() {
        let mut bus = MessageBus::new();
        assert!(!bus.remove_channel("ghost"));
    }

    #[test]
    fn test_channel_lookup() {
        let mut bus = MessageBus::new();
        bus.create_channel("risk", "risk.");
        let ch = bus.channel("risk").unwrap();
        assert_eq!(ch.name, "risk");
        assert_eq!(ch.topic_filter, "risk.");
    }

    #[test]
    fn test_channel_depth() {
        let mut bus = MessageBus::with_config(small_config());
        bus.create_channel("all", "");
        bus.emit("test", "sender", 1.0);
        assert_eq!(bus.channel_depth("all"), 1);
    }

    #[test]
    fn test_channel_names() {
        let mut bus = MessageBus::new();
        bus.create_channel("a", "");
        bus.create_channel("b", "");
        let mut names = bus.channel_names();
        names.sort();
        assert_eq!(names, vec!["a", "b"]);
    }

    // -------------------------------------------------------------------
    // Sending
    // -------------------------------------------------------------------

    #[test]
    fn test_send_returns_id() {
        let mut bus = MessageBus::new();
        bus.create_channel("all", "");
        let id1 = bus.emit("topic", "sender", 1.0);
        let id2 = bus.emit("topic", "sender", 2.0);
        assert!(id2 > id1);
    }

    #[test]
    fn test_send_routes_to_matching_channel() {
        let mut bus = MessageBus::new();
        bus.create_channel("risk", "risk.");
        bus.create_channel("market", "market.");

        bus.emit("risk.alert", "amygdala", 0.95);
        assert_eq!(bus.channel_depth("risk"), 1);
        assert_eq!(bus.channel_depth("market"), 0);
    }

    #[test]
    fn test_send_routes_to_multiple_channels() {
        let mut bus = MessageBus::new();
        bus.create_channel("all", "");
        bus.create_channel("risk", "risk.");

        bus.emit("risk.alert", "amygdala", 0.95);
        assert_eq!(bus.channel_depth("all"), 1);
        assert_eq!(bus.channel_depth("risk"), 1);
        assert_eq!(bus.stats().total_delivered, 2);
    }

    #[test]
    fn test_send_no_matching_channel() {
        let mut bus = MessageBus::new();
        bus.create_channel("risk", "risk.");
        bus.emit("market.tick", "thalamus", 100.0);
        assert_eq!(bus.stats().total_dropped, 1);
    }

    #[test]
    fn test_send_with_body() {
        let mut bus = MessageBus::new();
        bus.create_channel("all", "");
        bus.send("topic", "sender", 42.0, Some("hello".into()));
        let msgs = bus.drain("all");
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].body.as_deref(), Some("hello"));
    }

    #[test]
    fn test_send_buffer_overflow() {
        let overflow_config = MessageBusConfig {
            channel_buffer_size: 4,
            ..small_config()
        };
        let mut bus = MessageBus::with_config(overflow_config); // buffer = 4
        bus.create_channel("all", "");

        for i in 0..6 {
            bus.emit("test", "sender", i as f64);
        }
        assert_eq!(bus.channel_depth("all"), 4);
        let ch = bus.channel("all").unwrap();
        assert_eq!(ch.total_dropped, 2);
    }

    #[test]
    fn test_topic_counts() {
        let mut bus = MessageBus::new();
        bus.create_channel("all", "");
        bus.emit("market.tick", "thalamus", 1.0);
        bus.emit("market.tick", "thalamus", 2.0);
        bus.emit("risk.alert", "amygdala", 3.0);
        assert_eq!(bus.topic_count(), 2);
        assert_eq!(bus.topic_send_count("market.tick"), 2);
        assert_eq!(bus.topic_send_count("risk.alert"), 1);
        assert_eq!(bus.topic_send_count("unknown"), 0);
    }

    // -------------------------------------------------------------------
    // Receiving
    // -------------------------------------------------------------------

    #[test]
    fn test_drain_channel() {
        let mut bus = MessageBus::new();
        bus.create_channel("all", "");
        bus.emit("a", "sender", 1.0);
        bus.emit("b", "sender", 2.0);

        let msgs = bus.drain("all");
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].topic, "a");
        assert_eq!(msgs[1].topic, "b");
        assert_eq!(bus.channel_depth("all"), 0);
    }

    #[test]
    fn test_drain_nonexistent_channel() {
        let mut bus = MessageBus::new();
        let msgs = bus.drain("ghost");
        assert!(msgs.is_empty());
    }

    #[test]
    fn test_peek() {
        let mut bus = MessageBus::new();
        bus.create_channel("all", "");
        bus.emit("test", "sender", 42.0);
        let msg = bus.peek("all").unwrap();
        assert_eq!(msg.topic, "test");
        assert!((msg.payload - 42.0).abs() < 1e-10);
        // Peek should not consume
        assert_eq!(bus.channel_depth("all"), 1);
    }

    #[test]
    fn test_peek_empty() {
        let mut bus = MessageBus::new();
        bus.create_channel("all", "");
        assert!(bus.peek("all").is_none());
    }

    #[test]
    fn test_drain_all() {
        let mut bus = MessageBus::new();
        bus.create_channel("risk", "risk.");
        bus.create_channel("market", "market.");
        bus.create_channel("empty", "empty.");

        bus.emit("risk.alert", "amygdala", 1.0);
        bus.emit("market.tick", "thalamus", 2.0);

        let all = bus.drain_all();
        assert_eq!(all.len(), 2); // empty channel not included
        assert!(all.contains_key("risk"));
        assert!(all.contains_key("market"));
        assert!(!all.contains_key("empty"));
    }

    // -------------------------------------------------------------------
    // Tick & process
    // -------------------------------------------------------------------

    #[test]
    fn test_tick_increments() {
        let mut bus = MessageBus::new();
        bus.tick();
        bus.tick();
        assert_eq!(bus.current_tick(), 2);
    }

    #[test]
    fn test_process() {
        let mut bus = MessageBus::new();
        assert!(bus.process().is_ok());
        assert_eq!(bus.current_tick(), 1);
    }

    // -------------------------------------------------------------------
    // EMA diagnostics
    // -------------------------------------------------------------------

    #[test]
    fn test_ema_initialises_on_first_tick() {
        let mut bus = MessageBus::with_config(small_config());
        bus.create_channel("all", "");
        bus.emit("test", "sender", 1.0);
        bus.tick();
        assert!(bus.smoothed_send_rate() > 0.0);
        assert!(bus.smoothed_delivery_rate() > 0.0);
    }

    #[test]
    fn test_ema_blends_on_subsequent_ticks() {
        let mut bus = MessageBus::with_config(MessageBusConfig {
            ema_decay: 0.5,
            ..MessageBusConfig::default()
        });
        bus.create_channel("all", "");

        // Tick 1: send 10 messages
        for _ in 0..10 {
            bus.emit("test", "sender", 1.0);
        }
        bus.tick(); // ema_send = 10

        // Tick 2: send 0 messages
        bus.tick(); // ema_send = 0.5 * 0 + 0.5 * 10 = 5

        assert!((bus.smoothed_send_rate() - 5.0).abs() < 1e-10);
    }

    // -------------------------------------------------------------------
    // Windowed diagnostics
    // -------------------------------------------------------------------

    #[test]
    fn test_windowed_send_rate() {
        let mut bus = MessageBus::with_config(small_config());
        bus.create_channel("all", "");

        for _ in 0..3 {
            bus.emit("test", "sender", 1.0);
            bus.tick();
        }
        assert!((bus.windowed_send_rate() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_windowed_delivery_rate() {
        let mut bus = MessageBus::with_config(small_config());
        bus.create_channel("all", "");

        for _ in 0..3 {
            bus.emit("test", "sender", 1.0);
            bus.tick();
        }
        assert!(bus.windowed_delivery_rate() > 0.0);
    }

    #[test]
    fn test_windowed_empty() {
        let bus = MessageBus::new();
        assert!((bus.windowed_send_rate()).abs() < 1e-10);
        assert!((bus.windowed_delivery_rate()).abs() < 1e-10);
        assert!((bus.windowed_buffer_depth()).abs() < 1e-10);
        assert!((bus.windowed_drop_rate()).abs() < 1e-10);
    }

    #[test]
    fn test_windowed_drop_rate_no_channels() {
        let mut bus = MessageBus::with_config(small_config());
        // No channels → all messages dropped
        bus.emit("test", "sender", 1.0);
        bus.tick();
        assert!((bus.windowed_drop_rate() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_windowed_buffer_depth() {
        let mut bus = MessageBus::with_config(small_config());
        bus.create_channel("all", "");
        bus.emit("test", "sender", 1.0);
        bus.tick();
        assert!(bus.windowed_buffer_depth() > 0.0);
    }

    #[test]
    fn test_is_send_rate_increasing() {
        let mut bus = MessageBus::with_config(small_config());
        bus.create_channel("all", "");

        // First half: low rate
        for _ in 0..3 {
            bus.emit("test", "sender", 1.0);
            bus.tick();
        }
        // Second half: high rate
        for _ in 0..3 {
            for _ in 0..5 {
                bus.emit("test", "sender", 1.0);
            }
            bus.tick();
        }
        assert!(bus.is_send_rate_increasing());
    }

    #[test]
    fn test_is_send_rate_increasing_insufficient_data() {
        let mut bus = MessageBus::new();
        bus.tick();
        assert!(!bus.is_send_rate_increasing());
    }

    #[test]
    fn test_is_buffer_growing() {
        let mut bus = MessageBus::with_config(small_config());
        bus.create_channel("all", "");

        // First half: drain immediately
        for _ in 0..3 {
            bus.tick();
        }
        // Second half: accumulate
        for _ in 0..3 {
            bus.emit("test", "sender", 1.0);
            bus.emit("test", "sender", 2.0);
            bus.tick();
        }
        assert!(bus.is_buffer_growing());
    }

    #[test]
    fn test_is_buffer_growing_insufficient_data() {
        let mut bus = MessageBus::new();
        bus.tick();
        assert!(!bus.is_buffer_growing());
    }

    // -------------------------------------------------------------------
    // Reset
    // -------------------------------------------------------------------

    #[test]
    fn test_reset() {
        let mut bus = MessageBus::with_config(small_config());
        bus.create_channel("all", "");
        bus.emit("test", "sender", 1.0);
        bus.tick();

        bus.reset();
        assert_eq!(bus.channel_count(), 0);
        assert_eq!(bus.current_tick(), 0);
        assert_eq!(bus.stats().total_sent, 0);
        assert_eq!(bus.topic_count(), 0);
    }

    // -------------------------------------------------------------------
    // Full lifecycle
    // -------------------------------------------------------------------

    #[test]
    fn test_full_lifecycle() {
        let mut bus = MessageBus::with_config(small_config());

        // Create channels
        bus.create_channel("risk", "risk.");
        bus.create_channel("market", "market.");
        bus.create_channel("all", "");

        // Simulate several ticks
        for i in 0..5 {
            bus.emit("market.tick", "thalamus", i as f64 * 100.0);
            if i % 2 == 0 {
                bus.emit("risk.var", "amygdala", 0.05);
            }

            // Drain risk channel each tick
            let risk_msgs = bus.drain("risk");
            if i % 2 == 0 {
                assert_eq!(risk_msgs.len(), 1);
            } else {
                assert!(risk_msgs.is_empty());
            }

            bus.tick();
        }

        assert_eq!(bus.current_tick(), 5);
        assert!(bus.stats().total_sent > 0);
        assert!(bus.stats().total_delivered > 0);
        assert!(bus.smoothed_send_rate() > 0.0);
        assert!(bus.smoothed_delivery_rate() > 0.0);

        // Market channel should have accumulated messages
        let market_msgs = bus.drain("market");
        assert_eq!(market_msgs.len(), 5);
    }

    #[test]
    fn test_window_rolls() {
        let mut bus = MessageBus::with_config(small_config()); // window = 5
        for _ in 0..10 {
            bus.tick();
        }
        assert!(bus.recent.len() <= bus.config.window_size);
    }

    // -------------------------------------------------------------------
    // Channel internals
    // -------------------------------------------------------------------

    #[test]
    fn test_channel_accepts_empty_filter() {
        let ch = Channel::new("all", "", 8);
        assert!(ch.accepts("anything.here"));
    }

    #[test]
    fn test_channel_accepts_prefix_match() {
        let ch = Channel::new("risk", "risk.", 8);
        assert!(ch.accepts("risk.alert"));
        assert!(ch.accepts("risk."));
        assert!(!ch.accepts("market.tick"));
        assert!(!ch.accepts("riskfree"));
    }

    #[test]
    fn test_channel_fifo_order() {
        let mut ch = Channel::new("test", "", 8);
        for i in 0..3 {
            ch.enqueue(Message {
                id: i + 1,
                topic: format!("t{}", i),
                sender: "s".into(),
                payload: i as f64,
                body: None,
                tick: 0,
            });
        }
        let msgs = ch.drain();
        assert_eq!(msgs[0].id, 1);
        assert_eq!(msgs[1].id, 2);
        assert_eq!(msgs[2].id, 3);
    }

    #[test]
    fn test_channel_overflow_drops_oldest() {
        let mut ch = Channel::new("test", "", 2);
        for i in 0..4 {
            ch.enqueue(Message {
                id: i + 1,
                topic: "t".into(),
                sender: "s".into(),
                payload: i as f64,
                body: None,
                tick: 0,
            });
        }
        assert_eq!(ch.depth(), 2);
        assert_eq!(ch.total_dropped, 2);
        let msgs = ch.drain();
        // Should have the last two messages
        assert_eq!(msgs[0].id, 3);
        assert_eq!(msgs[1].id, 4);
    }

    #[test]
    fn test_channel_stats() {
        let mut ch = Channel::new("test", "", 4);
        for i in 0..3 {
            ch.enqueue(Message {
                id: i + 1,
                topic: "t".into(),
                sender: "s".into(),
                payload: 0.0,
                body: None,
                tick: 0,
            });
        }
        assert_eq!(ch.total_enqueued, 3);
        let _ = ch.drain();
        assert_eq!(ch.total_consumed, 3);
    }
}
