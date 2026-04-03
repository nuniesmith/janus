//! Event dispatching with typed events, subscriber registration, and priority
//! queues.
//!
//! Part of the Integration region — State component.
//!
//! `EventDispatcher` provides a centralised publish/subscribe mechanism for
//! inter-region communication. Events carry a topic, priority, and payload.
//! Subscribers register interest by topic pattern and receive dispatched events
//! in priority order. The module tracks EMA-smoothed dispatch rates and
//! windowed diagnostics.

use crate::common::Result;
use std::collections::{HashMap, VecDeque};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the event dispatcher.
#[derive(Debug, Clone)]
pub struct EventDispatcherConfig {
    /// Maximum number of events that can be buffered before oldest are dropped.
    pub max_queue_size: usize,
    /// Maximum number of subscribers that can be registered.
    pub max_subscribers: usize,
    /// Maximum number of topics that can be tracked.
    pub max_topics: usize,
    /// EMA decay factor (0 < α < 1). Higher → faster adaptation.
    pub ema_decay: f64,
    /// Window size for rolling diagnostics.
    pub window_size: usize,
}

impl Default for EventDispatcherConfig {
    fn default() -> Self {
        Self {
            max_queue_size: 4096,
            max_subscribers: 256,
            max_topics: 128,
            ema_decay: 0.1,
            window_size: 50,
        }
    }
}

// ---------------------------------------------------------------------------
// Event types
// ---------------------------------------------------------------------------

/// Priority level for events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EventPriority {
    /// Low priority — informational, best-effort delivery.
    Low = 0,
    /// Normal priority — standard operational events.
    Normal = 1,
    /// High priority — important state changes.
    High = 2,
    /// Critical priority — circuit breakers, emergency signals.
    Critical = 3,
}

/// A typed event flowing through the dispatcher.
#[derive(Debug, Clone)]
pub struct Event {
    /// Unique, monotonically increasing event identifier.
    pub id: u64,
    /// Topic string (e.g. `"risk.breach"`, `"market.tick"`).
    pub topic: String,
    /// Priority level.
    pub priority: EventPriority,
    /// Numeric payload (domain-specific meaning).
    pub payload: f64,
    /// Optional human-readable message.
    pub message: Option<String>,
    /// Tick at which the event was published.
    pub tick: u64,
}

/// Record of a dispatched event delivery for a single subscriber.
#[derive(Debug, Clone)]
pub struct Delivery {
    /// Event id that was delivered.
    pub event_id: u64,
    /// Subscriber id that received the event.
    pub subscriber_id: u64,
    /// Topic of the delivered event.
    pub topic: String,
}

// ---------------------------------------------------------------------------
// Subscriber
// ---------------------------------------------------------------------------

/// A registered subscriber.
#[derive(Debug, Clone)]
pub struct Subscriber {
    /// Unique subscriber identifier.
    pub id: u64,
    /// Human-readable name.
    pub name: String,
    /// Topic filter — only events whose topic starts with this prefix are
    /// delivered. An empty string matches all topics.
    pub topic_filter: String,
    /// Minimum priority level the subscriber is interested in.
    pub min_priority: EventPriority,
    /// Number of events delivered to this subscriber.
    pub delivery_count: u64,
}

impl Subscriber {
    /// Whether this subscriber is interested in a given event.
    fn matches(&self, event: &Event) -> bool {
        if event.priority < self.min_priority {
            return false;
        }
        if self.topic_filter.is_empty() {
            return true;
        }
        event.topic.starts_with(&self.topic_filter)
    }
}

// ---------------------------------------------------------------------------
// Tick snapshot (windowed diagnostics)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct TickSnapshot {
    events_published: u64,
    events_dispatched: u64,
    events_dropped: u64,
    deliveries: u64,
    queue_depth: usize,
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Cumulative statistics for the event dispatcher.
#[derive(Debug, Clone)]
pub struct EventDispatcherStats {
    /// Total number of events published.
    pub total_published: u64,
    /// Total number of events successfully dispatched (delivered to ≥1 subscriber).
    pub total_dispatched: u64,
    /// Total number of events dropped (no matching subscriber or queue overflow).
    pub total_dropped: u64,
    /// Total individual deliveries (one event × N subscribers = N deliveries).
    pub total_deliveries: u64,
    /// Total number of subscribers ever registered.
    pub total_subscriptions: u64,
    /// Total number of subscribers removed.
    pub total_unsubscriptions: u64,
    /// EMA-smoothed events-per-tick publish rate.
    pub ema_publish_rate: f64,
    /// EMA-smoothed deliveries-per-tick rate.
    pub ema_delivery_rate: f64,
    /// EMA-smoothed queue depth.
    pub ema_queue_depth: f64,
}

impl Default for EventDispatcherStats {
    fn default() -> Self {
        Self {
            total_published: 0,
            total_dispatched: 0,
            total_dropped: 0,
            total_deliveries: 0,
            total_subscriptions: 0,
            total_unsubscriptions: 0,
            ema_publish_rate: 0.0,
            ema_delivery_rate: 0.0,
            ema_queue_depth: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// EventDispatcher
// ---------------------------------------------------------------------------

/// Centralised publish/subscribe event dispatcher.
///
/// Provides:
/// * Topic-based event publishing with priority levels.
/// * Subscriber registration with topic-prefix filtering and minimum priority.
/// * Priority-ordered dispatch (highest priority first).
/// * Tick-based lifecycle with EMA and windowed diagnostics.
pub struct EventDispatcher {
    config: EventDispatcherConfig,
    /// Pending event queue, ordered by publish time (newest at back).
    queue: VecDeque<Event>,
    /// Registered subscribers keyed by id.
    subscribers: HashMap<u64, Subscriber>,
    /// Per-topic publish counts.
    topic_counts: HashMap<String, u64>,
    /// Monotonically increasing event id counter.
    next_event_id: u64,
    /// Monotonically increasing subscriber id counter.
    next_subscriber_id: u64,
    /// Current tick.
    tick: u64,
    /// Per-tick counters (reset each tick).
    tick_published: u64,
    tick_dispatched: u64,
    tick_dropped: u64,
    tick_deliveries: u64,
    /// Deliveries from the most recent `dispatch` call.
    last_deliveries: Vec<Delivery>,
    /// EMA initialisation flag.
    ema_initialized: bool,
    /// Rolling window of tick snapshots.
    recent: VecDeque<TickSnapshot>,
    /// Cumulative statistics.
    stats: EventDispatcherStats,
}

impl Default for EventDispatcher {
    fn default() -> Self {
        Self::new()
    }
}

impl EventDispatcher {
    // -------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------

    /// Create with default configuration.
    pub fn new() -> Self {
        Self::with_config(EventDispatcherConfig::default())
    }

    /// Create with explicit configuration.
    pub fn with_config(config: EventDispatcherConfig) -> Self {
        Self {
            queue: VecDeque::with_capacity(config.max_queue_size.min(256)),
            subscribers: HashMap::with_capacity(config.max_subscribers.min(32)),
            topic_counts: HashMap::with_capacity(config.max_topics.min(32)),
            next_event_id: 1,
            next_subscriber_id: 1,
            tick: 0,
            tick_published: 0,
            tick_dispatched: 0,
            tick_dropped: 0,
            tick_deliveries: 0,
            last_deliveries: Vec::new(),
            ema_initialized: false,
            recent: VecDeque::with_capacity(config.window_size),
            stats: EventDispatcherStats::default(),
            config,
        }
    }

    // -------------------------------------------------------------------
    // Subscriber management
    // -------------------------------------------------------------------

    /// Register a subscriber. Returns the subscriber id.
    ///
    /// `topic_filter` acts as a prefix match — an empty string matches all
    /// topics. `min_priority` sets the lowest priority the subscriber will
    /// receive.
    pub fn subscribe(
        &mut self,
        name: impl Into<String>,
        topic_filter: impl Into<String>,
        min_priority: EventPriority,
    ) -> u64 {
        if self.subscribers.len() >= self.config.max_subscribers {
            return 0; // capacity exceeded
        }
        let id = self.next_subscriber_id;
        self.next_subscriber_id += 1;
        self.subscribers.insert(
            id,
            Subscriber {
                id,
                name: name.into(),
                topic_filter: topic_filter.into(),
                min_priority,
                delivery_count: 0,
            },
        );
        self.stats.total_subscriptions += 1;
        id
    }

    /// Remove a subscriber by id. Returns `true` if found and removed.
    pub fn unsubscribe(&mut self, subscriber_id: u64) -> bool {
        let removed = self.subscribers.remove(&subscriber_id).is_some();
        if removed {
            self.stats.total_unsubscriptions += 1;
        }
        removed
    }

    /// Number of currently registered subscribers.
    pub fn subscriber_count(&self) -> usize {
        self.subscribers.len()
    }

    /// Get a subscriber record by id.
    pub fn subscriber(&self, id: u64) -> Option<&Subscriber> {
        self.subscribers.get(&id)
    }

    // -------------------------------------------------------------------
    // Publishing
    // -------------------------------------------------------------------

    /// Publish an event into the dispatch queue.
    ///
    /// If the queue is at capacity, the oldest (lowest-priority) event is
    /// dropped to make room.
    pub fn publish(
        &mut self,
        topic: impl Into<String>,
        priority: EventPriority,
        payload: f64,
        message: Option<String>,
    ) -> u64 {
        let topic = topic.into();
        let id = self.next_event_id;
        self.next_event_id += 1;

        // Track topic counts
        if self.topic_counts.len() < self.config.max_topics
            || self.topic_counts.contains_key(&topic)
        {
            *self.topic_counts.entry(topic.clone()).or_insert(0) += 1;
        }

        let event = Event {
            id,
            topic,
            priority,
            payload,
            message,
            tick: self.tick,
        };

        // Evict oldest if at capacity
        if self.queue.len() >= self.config.max_queue_size {
            self.queue.pop_front();
            self.stats.total_dropped += 1;
            self.tick_dropped += 1;
        }

        self.queue.push_back(event);
        self.stats.total_published += 1;
        self.tick_published += 1;

        id
    }

    /// Convenience: publish with only topic, priority, and payload.
    pub fn emit(&mut self, topic: impl Into<String>, priority: EventPriority, payload: f64) -> u64 {
        self.publish(topic, priority, payload, None)
    }

    /// Number of events currently in the queue.
    pub fn queue_depth(&self) -> usize {
        self.queue.len()
    }

    /// Number of distinct topics seen.
    pub fn topic_count(&self) -> usize {
        self.topic_counts.len()
    }

    /// Get the publish count for a specific topic.
    pub fn topic_publish_count(&self, topic: &str) -> u64 {
        self.topic_counts.get(topic).copied().unwrap_or(0)
    }

    // -------------------------------------------------------------------
    // Dispatching
    // -------------------------------------------------------------------

    /// Dispatch all queued events to matching subscribers.
    ///
    /// Events are sorted by priority (highest first) before delivery.
    /// Returns the number of individual deliveries made.
    pub fn dispatch(&mut self) -> usize {
        if self.queue.is_empty() || self.subscribers.is_empty() {
            let dropped = self.queue.len();
            self.stats.total_dropped += dropped as u64;
            self.tick_dropped += dropped as u64;
            self.queue.clear();
            self.last_deliveries.clear();
            return 0;
        }

        // Drain queue and sort by priority descending
        let mut events: Vec<Event> = self.queue.drain(..).collect();
        events.sort_by(|a, b| b.priority.cmp(&a.priority));

        let mut deliveries: Vec<Delivery> = Vec::new();
        let sub_ids: Vec<u64> = self.subscribers.keys().copied().collect();

        for event in &events {
            let mut matched = false;
            for &sid in &sub_ids {
                if let Some(sub) = self.subscribers.get(&sid) {
                    if sub.matches(event) {
                        deliveries.push(Delivery {
                            event_id: event.id,
                            subscriber_id: sid,
                            topic: event.topic.clone(),
                        });
                        matched = true;
                    }
                }
            }
            if matched {
                self.stats.total_dispatched += 1;
                self.tick_dispatched += 1;
            } else {
                self.stats.total_dropped += 1;
                self.tick_dropped += 1;
            }
        }

        // Update per-subscriber delivery counts
        for d in &deliveries {
            if let Some(sub) = self.subscribers.get_mut(&d.subscriber_id) {
                sub.delivery_count += 1;
            }
        }

        let count = deliveries.len();
        self.stats.total_deliveries += count as u64;
        self.tick_deliveries += count as u64;
        self.last_deliveries = deliveries;

        count
    }

    /// Get deliveries from the most recent `dispatch` call.
    pub fn last_deliveries(&self) -> &[Delivery] {
        &self.last_deliveries
    }

    // -------------------------------------------------------------------
    // Tick lifecycle
    // -------------------------------------------------------------------

    /// Advance the dispatcher tick and update diagnostics.
    ///
    /// Call once per system tick after publishing and dispatching.
    pub fn tick(&mut self) {
        self.tick += 1;

        let snapshot = TickSnapshot {
            events_published: self.tick_published,
            events_dispatched: self.tick_dispatched,
            events_dropped: self.tick_dropped,
            deliveries: self.tick_deliveries,
            queue_depth: self.queue.len(),
        };

        // EMA update
        let alpha = self.config.ema_decay;
        if !self.ema_initialized {
            self.stats.ema_publish_rate = snapshot.events_published as f64;
            self.stats.ema_delivery_rate = snapshot.deliveries as f64;
            self.stats.ema_queue_depth = snapshot.queue_depth as f64;
            self.ema_initialized = true;
        } else {
            self.stats.ema_publish_rate = alpha * snapshot.events_published as f64
                + (1.0 - alpha) * self.stats.ema_publish_rate;
            self.stats.ema_delivery_rate =
                alpha * snapshot.deliveries as f64 + (1.0 - alpha) * self.stats.ema_delivery_rate;
            self.stats.ema_queue_depth =
                alpha * snapshot.queue_depth as f64 + (1.0 - alpha) * self.stats.ema_queue_depth;
        }

        // Windowed
        if self.recent.len() >= self.config.window_size {
            self.recent.pop_front();
        }
        self.recent.push_back(snapshot);

        // Reset per-tick counters
        self.tick_published = 0;
        self.tick_dispatched = 0;
        self.tick_dropped = 0;
        self.tick_deliveries = 0;
    }

    /// Current tick value.
    pub fn current_tick(&self) -> u64 {
        self.tick
    }

    /// Main processing function: dispatch pending events then tick.
    pub fn process(&mut self) -> Result<()> {
        self.dispatch();
        self.tick();
        Ok(())
    }

    // -------------------------------------------------------------------
    // Statistics & diagnostics
    // -------------------------------------------------------------------

    /// Reference to cumulative statistics.
    pub fn stats(&self) -> &EventDispatcherStats {
        &self.stats
    }

    /// Reference to configuration.
    pub fn config(&self) -> &EventDispatcherConfig {
        &self.config
    }

    /// EMA-smoothed publish rate (events per tick).
    pub fn smoothed_publish_rate(&self) -> f64 {
        self.stats.ema_publish_rate
    }

    /// EMA-smoothed delivery rate (deliveries per tick).
    pub fn smoothed_delivery_rate(&self) -> f64 {
        self.stats.ema_delivery_rate
    }

    /// EMA-smoothed queue depth.
    pub fn smoothed_queue_depth(&self) -> f64 {
        self.stats.ema_queue_depth
    }

    /// Windowed average publish rate.
    pub fn windowed_publish_rate(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.recent.iter().map(|s| s.events_published as f64).sum();
        sum / self.recent.len() as f64
    }

    /// Windowed average delivery rate.
    pub fn windowed_delivery_rate(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.recent.iter().map(|s| s.deliveries as f64).sum();
        sum / self.recent.len() as f64
    }

    /// Windowed average queue depth.
    pub fn windowed_queue_depth(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.recent.iter().map(|s| s.queue_depth as f64).sum();
        sum / self.recent.len() as f64
    }

    /// Windowed drop rate (fraction of published events that were dropped).
    pub fn windowed_drop_rate(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let total_pub: u64 = self.recent.iter().map(|s| s.events_published).sum();
        let total_drop: u64 = self.recent.iter().map(|s| s.events_dropped).sum();
        if total_pub == 0 {
            return 0.0;
        }
        total_drop as f64 / total_pub as f64
    }

    /// Whether the publish rate is trending upward over the window.
    pub fn is_publish_rate_increasing(&self) -> bool {
        if self.recent.len() < 4 {
            return false;
        }
        let n = self.recent.len();
        let half = n / 2;
        let first_avg: f64 = self
            .recent
            .iter()
            .take(half)
            .map(|s| s.events_published as f64)
            .sum::<f64>()
            / half as f64;
        let second_avg: f64 = self
            .recent
            .iter()
            .skip(half)
            .map(|s| s.events_published as f64)
            .sum::<f64>()
            / (n - half) as f64;
        second_avg > first_avg
    }

    /// Clear the event queue without dispatching.
    pub fn clear_queue(&mut self) {
        let dropped = self.queue.len();
        self.stats.total_dropped += dropped as u64;
        self.tick_dropped += dropped as u64;
        self.queue.clear();
    }

    /// Reset all state.
    pub fn reset(&mut self) {
        self.queue.clear();
        self.subscribers.clear();
        self.topic_counts.clear();
        self.next_event_id = 1;
        self.next_subscriber_id = 1;
        self.tick = 0;
        self.tick_published = 0;
        self.tick_dispatched = 0;
        self.tick_dropped = 0;
        self.tick_deliveries = 0;
        self.last_deliveries.clear();
        self.ema_initialized = false;
        self.recent.clear();
        self.stats = EventDispatcherStats::default();
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> EventDispatcherConfig {
        EventDispatcherConfig {
            max_queue_size: 8,
            max_subscribers: 4,
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
        let ed = EventDispatcher::new();
        assert_eq!(ed.queue_depth(), 0);
        assert_eq!(ed.subscriber_count(), 0);
        assert_eq!(ed.current_tick(), 0);
    }

    #[test]
    fn test_with_config() {
        let ed = EventDispatcher::with_config(small_config());
        assert_eq!(ed.config().max_queue_size, 8);
        assert_eq!(ed.config().max_subscribers, 4);
    }

    // -------------------------------------------------------------------
    // Subscriber management
    // -------------------------------------------------------------------

    #[test]
    fn test_subscribe() {
        let mut ed = EventDispatcher::with_config(small_config());
        let id = ed.subscribe("risk_monitor", "risk.", EventPriority::Normal);
        assert!(id > 0);
        assert_eq!(ed.subscriber_count(), 1);
    }

    #[test]
    fn test_subscribe_at_capacity() {
        let mut ed = EventDispatcher::with_config(small_config()); // max = 4
        for i in 0..4 {
            let id = ed.subscribe(format!("s{}", i), "", EventPriority::Low);
            assert!(id > 0);
        }
        let id = ed.subscribe("overflow", "", EventPriority::Low);
        assert_eq!(id, 0); // rejected
        assert_eq!(ed.subscriber_count(), 4);
    }

    #[test]
    fn test_unsubscribe() {
        let mut ed = EventDispatcher::with_config(small_config());
        let id = ed.subscribe("s1", "", EventPriority::Low);
        assert!(ed.unsubscribe(id));
        assert_eq!(ed.subscriber_count(), 0);
        assert_eq!(ed.stats().total_unsubscriptions, 1);
    }

    #[test]
    fn test_unsubscribe_nonexistent() {
        let mut ed = EventDispatcher::new();
        assert!(!ed.unsubscribe(999));
    }

    #[test]
    fn test_subscriber_lookup() {
        let mut ed = EventDispatcher::new();
        let id = ed.subscribe("monitor", "risk.", EventPriority::High);
        let sub = ed.subscriber(id).unwrap();
        assert_eq!(sub.name, "monitor");
        assert_eq!(sub.topic_filter, "risk.");
        assert_eq!(sub.min_priority, EventPriority::High);
    }

    // -------------------------------------------------------------------
    // Publishing
    // -------------------------------------------------------------------

    #[test]
    fn test_publish_returns_id() {
        let mut ed = EventDispatcher::new();
        let id1 = ed.emit("market.tick", EventPriority::Normal, 100.0);
        let id2 = ed.emit("market.tick", EventPriority::Normal, 101.0);
        assert!(id2 > id1);
    }

    #[test]
    fn test_publish_queues_event() {
        let mut ed = EventDispatcher::new();
        ed.emit("market.tick", EventPriority::Normal, 100.0);
        assert_eq!(ed.queue_depth(), 1);
        assert_eq!(ed.stats().total_published, 1);
    }

    #[test]
    fn test_publish_with_message() {
        let mut ed = EventDispatcher::new();
        ed.publish(
            "risk.breach",
            EventPriority::Critical,
            0.95,
            Some("VaR limit exceeded".into()),
        );
        assert_eq!(ed.queue_depth(), 1);
    }

    #[test]
    fn test_publish_evicts_at_capacity() {
        let mut ed = EventDispatcher::with_config(small_config()); // max_queue = 8
        for i in 0..10 {
            ed.emit("test", EventPriority::Normal, i as f64);
        }
        assert!(ed.queue_depth() <= 8);
        assert!(ed.stats().total_dropped > 0);
    }

    #[test]
    fn test_topic_counts() {
        let mut ed = EventDispatcher::new();
        ed.emit("market.tick", EventPriority::Normal, 1.0);
        ed.emit("market.tick", EventPriority::Normal, 2.0);
        ed.emit("risk.breach", EventPriority::High, 3.0);
        assert_eq!(ed.topic_count(), 2);
        assert_eq!(ed.topic_publish_count("market.tick"), 2);
        assert_eq!(ed.topic_publish_count("risk.breach"), 1);
        assert_eq!(ed.topic_publish_count("unknown"), 0);
    }

    // -------------------------------------------------------------------
    // Dispatching
    // -------------------------------------------------------------------

    #[test]
    fn test_dispatch_to_matching_subscriber() {
        let mut ed = EventDispatcher::new();
        ed.subscribe("all", "", EventPriority::Low);
        ed.emit("market.tick", EventPriority::Normal, 100.0);
        let count = ed.dispatch();
        assert_eq!(count, 1);
        assert_eq!(ed.stats().total_dispatched, 1);
        assert_eq!(ed.stats().total_deliveries, 1);
        assert_eq!(ed.queue_depth(), 0);
    }

    #[test]
    fn test_dispatch_topic_filter() {
        let mut ed = EventDispatcher::new();
        let risk_sub = ed.subscribe("risk_only", "risk.", EventPriority::Low);
        ed.subscribe("market_only", "market.", EventPriority::Low);

        ed.emit("risk.breach", EventPriority::High, 0.95);
        let count = ed.dispatch();
        assert_eq!(count, 1); // only risk subscriber matches

        let sub = ed.subscriber(risk_sub).unwrap();
        assert_eq!(sub.delivery_count, 1);
    }

    #[test]
    fn test_dispatch_priority_filter() {
        let mut ed = EventDispatcher::new();
        ed.subscribe("critical_only", "", EventPriority::Critical);
        ed.emit("test", EventPriority::Normal, 1.0);
        let count = ed.dispatch();
        assert_eq!(count, 0); // event priority too low
        assert_eq!(ed.stats().total_dropped, 1);
    }

    #[test]
    fn test_dispatch_multiple_subscribers() {
        let mut ed = EventDispatcher::new();
        ed.subscribe("s1", "", EventPriority::Low);
        ed.subscribe("s2", "", EventPriority::Low);
        ed.emit("test", EventPriority::Normal, 1.0);
        let count = ed.dispatch();
        assert_eq!(count, 2); // both subscribers match
        assert_eq!(ed.stats().total_dispatched, 1); // one event dispatched
        assert_eq!(ed.stats().total_deliveries, 2); // two deliveries
    }

    #[test]
    fn test_dispatch_empty_queue() {
        let mut ed = EventDispatcher::new();
        ed.subscribe("s1", "", EventPriority::Low);
        let count = ed.dispatch();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_dispatch_no_subscribers() {
        let mut ed = EventDispatcher::new();
        ed.emit("test", EventPriority::Normal, 1.0);
        let count = ed.dispatch();
        assert_eq!(count, 0);
        assert_eq!(ed.stats().total_dropped, 1);
    }

    #[test]
    fn test_dispatch_priority_order() {
        let mut ed = EventDispatcher::new();
        ed.subscribe("all", "", EventPriority::Low);

        ed.emit("low", EventPriority::Low, 1.0);
        ed.emit("critical", EventPriority::Critical, 2.0);
        ed.emit("normal", EventPriority::Normal, 3.0);

        ed.dispatch();
        let deliveries = ed.last_deliveries();
        assert_eq!(deliveries.len(), 3);
        // Critical should come first
        assert_eq!(deliveries[0].topic, "critical");
        assert_eq!(deliveries[1].topic, "normal");
        assert_eq!(deliveries[2].topic, "low");
    }

    #[test]
    fn test_last_deliveries_cleared_on_next_dispatch() {
        let mut ed = EventDispatcher::new();
        ed.subscribe("all", "", EventPriority::Low);
        ed.emit("test", EventPriority::Normal, 1.0);
        ed.dispatch();
        assert_eq!(ed.last_deliveries().len(), 1);

        // Dispatch again with empty queue
        ed.dispatch();
        assert_eq!(ed.last_deliveries().len(), 0);
    }

    // -------------------------------------------------------------------
    // Tick & process
    // -------------------------------------------------------------------

    #[test]
    fn test_tick_increments() {
        let mut ed = EventDispatcher::new();
        ed.tick();
        ed.tick();
        assert_eq!(ed.current_tick(), 2);
    }

    #[test]
    fn test_process() {
        let mut ed = EventDispatcher::new();
        ed.subscribe("all", "", EventPriority::Low);
        ed.emit("test", EventPriority::Normal, 1.0);
        assert!(ed.process().is_ok());
        assert_eq!(ed.current_tick(), 1);
        assert_eq!(ed.queue_depth(), 0); // dispatched
    }

    // -------------------------------------------------------------------
    // EMA diagnostics
    // -------------------------------------------------------------------

    #[test]
    fn test_ema_initialises_on_first_tick() {
        let mut ed = EventDispatcher::with_config(small_config());
        ed.subscribe("all", "", EventPriority::Low);
        ed.emit("test", EventPriority::Normal, 1.0);
        ed.dispatch();
        ed.tick();
        assert!(ed.smoothed_publish_rate() > 0.0);
        assert!(ed.smoothed_delivery_rate() > 0.0);
    }

    #[test]
    fn test_ema_blends_on_subsequent_ticks() {
        let mut ed = EventDispatcher::with_config(EventDispatcherConfig {
            ema_decay: 0.5,
            ..EventDispatcherConfig::default()
        });
        ed.subscribe("all", "", EventPriority::Low);

        // Tick 1: publish 10 events
        for _ in 0..10 {
            ed.emit("test", EventPriority::Normal, 1.0);
        }
        ed.dispatch();
        ed.tick(); // ema_publish = 10

        // Tick 2: publish 0 events
        ed.tick(); // ema_publish = 0.5 * 0 + 0.5 * 10 = 5

        assert!((ed.smoothed_publish_rate() - 5.0).abs() < 1e-10);
    }

    // -------------------------------------------------------------------
    // Windowed diagnostics
    // -------------------------------------------------------------------

    #[test]
    fn test_windowed_publish_rate() {
        let mut ed = EventDispatcher::with_config(small_config());
        ed.subscribe("all", "", EventPriority::Low);

        for _ in 0..3 {
            ed.emit("test", EventPriority::Normal, 1.0);
            ed.dispatch();
            ed.tick();
        }
        assert!((ed.windowed_publish_rate() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_windowed_delivery_rate() {
        let mut ed = EventDispatcher::with_config(small_config());
        ed.subscribe("all", "", EventPriority::Low);

        for _ in 0..3 {
            ed.emit("test", EventPriority::Normal, 1.0);
            ed.dispatch();
            ed.tick();
        }
        assert!(ed.windowed_delivery_rate() > 0.0);
    }

    #[test]
    fn test_windowed_empty() {
        let ed = EventDispatcher::new();
        assert!((ed.windowed_publish_rate()).abs() < 1e-10);
        assert!((ed.windowed_delivery_rate()).abs() < 1e-10);
        assert!((ed.windowed_queue_depth()).abs() < 1e-10);
        assert!((ed.windowed_drop_rate()).abs() < 1e-10);
    }

    #[test]
    fn test_windowed_drop_rate() {
        let mut ed = EventDispatcher::with_config(small_config());
        // No subscribers → all published events are dropped on dispatch
        ed.emit("test", EventPriority::Normal, 1.0);
        ed.dispatch();
        ed.tick();
        // 1 published, 1 dropped → drop rate = 1.0
        assert!((ed.windowed_drop_rate() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_is_publish_rate_increasing() {
        let mut ed = EventDispatcher::with_config(small_config());
        ed.subscribe("all", "", EventPriority::Low);

        // First half: low publish rate
        for _ in 0..3 {
            ed.emit("test", EventPriority::Normal, 1.0);
            ed.dispatch();
            ed.tick();
        }
        // Second half: high publish rate
        for _ in 0..3 {
            for _ in 0..5 {
                ed.emit("test", EventPriority::Normal, 1.0);
            }
            ed.dispatch();
            ed.tick();
        }
        assert!(ed.is_publish_rate_increasing());
    }

    #[test]
    fn test_is_publish_rate_increasing_insufficient_data() {
        let mut ed = EventDispatcher::new();
        ed.tick();
        assert!(!ed.is_publish_rate_increasing());
    }

    // -------------------------------------------------------------------
    // Queue management
    // -------------------------------------------------------------------

    #[test]
    fn test_clear_queue() {
        let mut ed = EventDispatcher::new();
        ed.emit("test", EventPriority::Normal, 1.0);
        ed.emit("test", EventPriority::Normal, 2.0);
        assert_eq!(ed.queue_depth(), 2);
        ed.clear_queue();
        assert_eq!(ed.queue_depth(), 0);
        assert_eq!(ed.stats().total_dropped, 2);
    }

    // -------------------------------------------------------------------
    // Reset
    // -------------------------------------------------------------------

    #[test]
    fn test_reset() {
        let mut ed = EventDispatcher::with_config(small_config());
        ed.subscribe("s1", "", EventPriority::Low);
        ed.emit("test", EventPriority::Normal, 1.0);
        ed.dispatch();
        ed.tick();

        ed.reset();
        assert_eq!(ed.queue_depth(), 0);
        assert_eq!(ed.subscriber_count(), 0);
        assert_eq!(ed.current_tick(), 0);
        assert_eq!(ed.stats().total_published, 0);
        assert_eq!(ed.topic_count(), 0);
    }

    // -------------------------------------------------------------------
    // Full lifecycle
    // -------------------------------------------------------------------

    #[test]
    fn test_full_lifecycle() {
        let mut ed = EventDispatcher::with_config(small_config());

        // Register subscribers
        let risk_sub = ed.subscribe("risk_monitor", "risk.", EventPriority::Normal);
        let _all_sub = ed.subscribe("logger", "", EventPriority::Low);

        // Simulate several ticks
        for i in 0..5 {
            ed.emit("market.tick", EventPriority::Normal, i as f64 * 100.0);
            if i % 2 == 0 {
                ed.emit("risk.var", EventPriority::High, 0.05);
            }
            ed.dispatch();
            ed.tick();
        }

        assert_eq!(ed.current_tick(), 5);
        assert!(ed.stats().total_published > 0);
        assert!(ed.stats().total_dispatched > 0);
        assert!(ed.stats().total_deliveries > 0);
        assert!(ed.smoothed_publish_rate() > 0.0);
        assert!(ed.smoothed_delivery_rate() > 0.0);

        // Risk subscriber should have received only risk events
        let risk = ed.subscriber(risk_sub).unwrap();
        assert!(risk.delivery_count > 0);
        // It should have fewer deliveries than the "all" subscriber
        assert!(risk.delivery_count <= ed.stats().total_deliveries);
    }

    #[test]
    fn test_window_rolls() {
        let mut ed = EventDispatcher::with_config(small_config()); // window = 5
        for _ in 0..10 {
            ed.tick();
        }
        assert!(ed.recent.len() <= ed.config.window_size);
    }

    // -------------------------------------------------------------------
    // Edge cases
    // -------------------------------------------------------------------

    #[test]
    fn test_emit_convenience() {
        let mut ed = EventDispatcher::new();
        let id = ed.emit("topic", EventPriority::Low, 42.0);
        assert!(id > 0);
        assert_eq!(ed.queue_depth(), 1);
    }

    #[test]
    fn test_subscriber_matches_empty_filter() {
        let sub = Subscriber {
            id: 1,
            name: "all".into(),
            topic_filter: "".into(),
            min_priority: EventPriority::Low,
            delivery_count: 0,
        };
        let event = Event {
            id: 1,
            topic: "anything.here".into(),
            priority: EventPriority::Normal,
            payload: 0.0,
            message: None,
            tick: 0,
        };
        assert!(sub.matches(&event));
    }

    #[test]
    fn test_subscriber_rejects_low_priority() {
        let sub = Subscriber {
            id: 1,
            name: "critical".into(),
            topic_filter: "".into(),
            min_priority: EventPriority::Critical,
            delivery_count: 0,
        };
        let event = Event {
            id: 1,
            topic: "test".into(),
            priority: EventPriority::High,
            payload: 0.0,
            message: None,
            tick: 0,
        };
        assert!(!sub.matches(&event));
    }

    #[test]
    fn test_subscriber_prefix_match() {
        let sub = Subscriber {
            id: 1,
            name: "risk".into(),
            topic_filter: "risk.".into(),
            min_priority: EventPriority::Low,
            delivery_count: 0,
        };
        let match_event = Event {
            id: 1,
            topic: "risk.breach".into(),
            priority: EventPriority::Normal,
            payload: 0.0,
            message: None,
            tick: 0,
        };
        let no_match = Event {
            id: 2,
            topic: "market.tick".into(),
            priority: EventPriority::Normal,
            payload: 0.0,
            message: None,
            tick: 0,
        };
        assert!(sub.matches(&match_event));
        assert!(!sub.matches(&no_match));
    }
}
