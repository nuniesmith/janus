//! WebSocket streaming
//!
//! Part of the Integration region — API component.
//!
//! `Websocket` manages WebSocket sessions for real-time streaming between the
//! neuromorphic system and external clients. It tracks active connections,
//! buffers outbound messages per session, monitors throughput, and exposes
//! EMA-smoothed and windowed diagnostics.

use crate::common::{Error, Result};
use std::collections::{HashMap, VecDeque};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the WebSocket manager.
#[derive(Debug, Clone)]
pub struct WebsocketConfig {
    /// Maximum number of concurrent sessions.
    pub max_sessions: usize,
    /// Maximum number of messages buffered per session before oldest are
    /// dropped.
    pub session_buffer_size: usize,
    /// Maximum message payload size in bytes.
    pub max_message_size: usize,
    /// Ticks after which an idle session is considered stale.
    pub idle_timeout_ticks: u64,
    /// EMA decay factor (0 < α < 1). Higher → faster adaptation.
    pub ema_decay: f64,
    /// Window size for rolling diagnostics.
    pub window_size: usize,
}

impl Default for WebsocketConfig {
    fn default() -> Self {
        Self {
            max_sessions: 256,
            session_buffer_size: 512,
            max_message_size: 65536,
            idle_timeout_ticks: 100,
            ema_decay: 0.1,
            window_size: 50,
        }
    }
}

// ---------------------------------------------------------------------------
// Message types
// ---------------------------------------------------------------------------

/// Type of a WebSocket frame / message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MessageType {
    /// UTF-8 text frame.
    Text,
    /// Binary frame.
    Binary,
    /// Ping control frame.
    Ping,
    /// Pong control frame.
    Pong,
    /// Close control frame.
    Close,
}

impl std::fmt::Display for MessageType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MessageType::Text => write!(f, "Text"),
            MessageType::Binary => write!(f, "Binary"),
            MessageType::Ping => write!(f, "Ping"),
            MessageType::Pong => write!(f, "Pong"),
            MessageType::Close => write!(f, "Close"),
        }
    }
}

// ---------------------------------------------------------------------------
// WebSocket message
// ---------------------------------------------------------------------------

/// A WebSocket message.
#[derive(Debug, Clone)]
pub struct WsMessage {
    /// Unique, monotonically increasing message identifier.
    pub id: u64,
    /// Message type.
    pub msg_type: MessageType,
    /// Topic / channel the message belongs to (application-level routing).
    pub topic: String,
    /// Payload body.
    pub payload: String,
    /// Size of the payload in bytes.
    pub size_bytes: usize,
    /// Tick at which the message was created.
    pub tick: u64,
}

// ---------------------------------------------------------------------------
// Session state
// ---------------------------------------------------------------------------

/// State of a WebSocket session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionState {
    /// Session is being established (handshake).
    Connecting,
    /// Session is open and active.
    Open,
    /// Session is in the process of closing.
    Closing,
    /// Session has been closed.
    Closed,
}

impl std::fmt::Display for SessionState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SessionState::Connecting => write!(f, "Connecting"),
            SessionState::Open => write!(f, "Open"),
            SessionState::Closing => write!(f, "Closing"),
            SessionState::Closed => write!(f, "Closed"),
        }
    }
}

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

/// A WebSocket session.
#[derive(Debug, Clone)]
pub struct Session {
    /// Unique session identifier.
    pub id: String,
    /// Client address or label.
    pub client: String,
    /// Current session state.
    pub state: SessionState,
    /// Subscribed topics (empty = all topics).
    pub subscriptions: Vec<String>,
    /// Outbound message buffer.
    buffer: VecDeque<WsMessage>,
    /// Maximum buffer capacity.
    buffer_capacity: usize,
    /// Tick at which the session was opened.
    pub opened_at_tick: u64,
    /// Tick of the most recent activity (send or receive).
    pub last_activity_tick: u64,
    /// Total messages sent to this session.
    pub total_sent: u64,
    /// Total messages received from this session.
    pub total_received: u64,
    /// Total messages dropped due to buffer overflow.
    pub total_dropped: u64,
    /// Total bytes sent.
    pub total_bytes_sent: u64,
    /// Total bytes received.
    pub total_bytes_received: u64,
}

impl Session {
    fn new(
        id: impl Into<String>,
        client: impl Into<String>,
        buffer_capacity: usize,
        tick: u64,
    ) -> Self {
        Self {
            id: id.into(),
            client: client.into(),
            state: SessionState::Connecting,
            subscriptions: Vec::new(),
            buffer: VecDeque::with_capacity(buffer_capacity.min(256)),
            buffer_capacity,
            opened_at_tick: tick,
            last_activity_tick: tick,
            total_sent: 0,
            total_received: 0,
            total_dropped: 0,
            total_bytes_sent: 0,
            total_bytes_received: 0,
        }
    }

    /// Whether the session accepts messages on the given topic.
    fn accepts_topic(&self, topic: &str) -> bool {
        if self.subscriptions.is_empty() {
            return true; // subscribed to everything
        }
        self.subscriptions.iter().any(|s| topic.starts_with(s))
    }

    /// Enqueue a message for outbound delivery. Drops the oldest if at
    /// capacity.
    fn enqueue(&mut self, msg: WsMessage) -> bool {
        let dropped = if self.buffer.len() >= self.buffer_capacity {
            self.buffer.pop_front();
            self.total_dropped += 1;
            true
        } else {
            false
        };
        self.total_bytes_sent += msg.size_bytes as u64;
        self.total_sent += 1;
        self.buffer.push_back(msg);
        dropped
    }

    /// Drain all buffered messages.
    fn drain_buffer(&mut self) -> Vec<WsMessage> {
        self.buffer.drain(..).collect()
    }

    /// Number of messages currently buffered.
    fn buffered_count(&self) -> usize {
        self.buffer.len()
    }
}

// ---------------------------------------------------------------------------
// Tick snapshot (windowed diagnostics)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct TickSnapshot {
    active_sessions: usize,
    messages_sent: u64,
    messages_received: u64,
    messages_dropped: u64,
    bytes_sent: u64,
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Cumulative statistics for the WebSocket manager.
#[derive(Debug, Clone)]
pub struct WebsocketStats {
    /// Total sessions ever opened.
    pub total_sessions_opened: u64,
    /// Total sessions closed.
    pub total_sessions_closed: u64,
    /// Total messages sent across all sessions.
    pub total_messages_sent: u64,
    /// Total messages received across all sessions.
    pub total_messages_received: u64,
    /// Total messages dropped across all sessions.
    pub total_messages_dropped: u64,
    /// Total bytes sent.
    pub total_bytes_sent: u64,
    /// Total bytes received.
    pub total_bytes_received: u64,
    /// Total sessions timed out due to inactivity.
    pub total_timeouts: u64,
    /// EMA-smoothed active session count.
    pub ema_active_sessions: f64,
    /// EMA-smoothed messages-sent per tick.
    pub ema_messages_per_tick: f64,
    /// EMA-smoothed bytes-sent per tick.
    pub ema_bytes_per_tick: f64,
}

impl Default for WebsocketStats {
    fn default() -> Self {
        Self {
            total_sessions_opened: 0,
            total_sessions_closed: 0,
            total_messages_sent: 0,
            total_messages_received: 0,
            total_messages_dropped: 0,
            total_bytes_sent: 0,
            total_bytes_received: 0,
            total_timeouts: 0,
            ema_active_sessions: 0.0,
            ema_messages_per_tick: 0.0,
            ema_bytes_per_tick: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Websocket manager
// ---------------------------------------------------------------------------

/// WebSocket session manager.
///
/// Manages concurrent sessions, routes messages by topic subscription,
/// tracks per-session and aggregate throughput, and provides EMA + windowed
/// diagnostics.
pub struct Websocket {
    config: WebsocketConfig,
    /// Active sessions keyed by session ID.
    sessions: HashMap<String, Session>,
    /// Global monotonically increasing message ID.
    next_message_id: u64,
    /// Current tick counter.
    tick: u64,
    /// Messages sent in the current tick (for EMA).
    sent_this_tick: u64,
    /// Bytes sent in the current tick (for EMA).
    bytes_this_tick: u64,
    /// Messages received in the current tick (for EMA).
    received_this_tick: u64,
    /// Messages dropped in the current tick.
    dropped_this_tick: u64,
    /// Whether EMA has been initialised.
    ema_initialized: bool,
    /// Rolling window of tick snapshots.
    recent: VecDeque<TickSnapshot>,
    /// Cumulative statistics.
    stats: WebsocketStats,
}

impl Default for Websocket {
    fn default() -> Self {
        Self::new()
    }
}

impl Websocket {
    // -------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------

    /// Create a new WebSocket manager with default configuration.
    pub fn new() -> Self {
        Self::with_config(WebsocketConfig::default())
    }

    /// Create a new WebSocket manager with the given configuration.
    pub fn with_config(config: WebsocketConfig) -> Self {
        Self {
            sessions: HashMap::new(),
            next_message_id: 1,
            tick: 0,
            sent_this_tick: 0,
            bytes_this_tick: 0,
            received_this_tick: 0,
            dropped_this_tick: 0,
            ema_initialized: false,
            recent: VecDeque::with_capacity(config.window_size.min(256)),
            stats: WebsocketStats::default(),
            config,
        }
    }

    // -------------------------------------------------------------------
    // Session management
    // -------------------------------------------------------------------

    /// Open a new session. Returns an error if the session ID already exists
    /// or the maximum number of sessions has been reached.
    pub fn open_session(
        &mut self,
        id: impl Into<String>,
        client: impl Into<String>,
    ) -> Result<()> {
        let id = id.into();
        if self.sessions.contains_key(&id) {
            return Err(Error::Configuration(format!(
                "Session '{}' already exists",
                id
            )));
        }
        if self.sessions.len() >= self.config.max_sessions {
            return Err(Error::Configuration(format!(
                "Maximum session count ({}) reached",
                self.config.max_sessions
            )));
        }

        let mut session = Session::new(&id, client, self.config.session_buffer_size, self.tick);
        session.state = SessionState::Open;
        self.sessions.insert(id, session);
        self.stats.total_sessions_opened += 1;
        Ok(())
    }

    /// Close a session by ID. The session transitions to `Closed` and is
    /// removed from the active set.
    pub fn close_session(&mut self, id: &str) -> Result<()> {
        let session = self
            .sessions
            .get_mut(id)
            .ok_or_else(|| Error::Configuration(format!("Unknown session '{}'", id)))?;
        session.state = SessionState::Closed;
        self.sessions.remove(id);
        self.stats.total_sessions_closed += 1;
        Ok(())
    }

    /// Subscribe a session to one or more topics.
    pub fn subscribe(&mut self, session_id: &str, topics: Vec<String>) -> Result<()> {
        let session = self
            .sessions
            .get_mut(session_id)
            .ok_or_else(|| Error::Configuration(format!("Unknown session '{}'", session_id)))?;
        for topic in topics {
            if !session.subscriptions.contains(&topic) {
                session.subscriptions.push(topic);
            }
        }
        Ok(())
    }

    /// Unsubscribe a session from one or more topics.
    pub fn unsubscribe(&mut self, session_id: &str, topics: &[String]) -> Result<()> {
        let session = self
            .sessions
            .get_mut(session_id)
            .ok_or_else(|| Error::Configuration(format!("Unknown session '{}'", session_id)))?;
        session.subscriptions.retain(|t| !topics.contains(t));
        Ok(())
    }

    /// Look up a session by ID.
    pub fn session(&self, id: &str) -> Option<&Session> {
        self.sessions.get(id)
    }

    /// Number of currently active sessions.
    pub fn active_session_count(&self) -> usize {
        self.sessions
            .values()
            .filter(|s| s.state == SessionState::Open)
            .count()
    }

    /// Total number of sessions (including those in non-open states still
    /// tracked).
    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }

    /// Return the IDs of all active sessions.
    pub fn session_ids(&self) -> Vec<&str> {
        self.sessions.keys().map(|s| s.as_str()).collect()
    }

    // -------------------------------------------------------------------
    // Sending messages
    // -------------------------------------------------------------------

    /// Broadcast a message to all sessions subscribed to the given topic.
    ///
    /// Returns the number of sessions the message was delivered to.
    pub fn broadcast(
        &mut self,
        topic: impl Into<String>,
        payload: impl Into<String>,
        msg_type: MessageType,
    ) -> Result<usize> {
        let topic = topic.into();
        let payload = payload.into();
        let size_bytes = payload.len();

        if size_bytes > self.config.max_message_size {
            return Err(Error::Configuration(format!(
                "Message size ({} bytes) exceeds max ({})",
                size_bytes, self.config.max_message_size
            )));
        }

        let tick = self.tick;
        let mut delivered = 0usize;
        let mut total_dropped = 0u64;

        // Collect session IDs that accept this topic.
        let matching_ids: Vec<String> = self
            .sessions
            .values()
            .filter(|s| s.state == SessionState::Open && s.accepts_topic(&topic))
            .map(|s| s.id.clone())
            .collect();

        for sid in &matching_ids {
            let msg_id = self.next_message_id;
            self.next_message_id += 1;

            let msg = WsMessage {
                id: msg_id,
                msg_type,
                topic: topic.clone(),
                payload: payload.clone(),
                size_bytes,
                tick,
            };

            if let Some(session) = self.sessions.get_mut(sid.as_str()) {
                let was_dropped = session.enqueue(msg);
                session.last_activity_tick = tick;
                delivered += 1;
                if was_dropped {
                    total_dropped += 1;
                }
            }
        }

        self.sent_this_tick += delivered as u64;
        self.bytes_this_tick += (size_bytes as u64) * (delivered as u64);
        self.dropped_this_tick += total_dropped;
        self.stats.total_messages_sent += delivered as u64;
        self.stats.total_bytes_sent += (size_bytes as u64) * (delivered as u64);
        self.stats.total_messages_dropped += total_dropped;

        Ok(delivered)
    }

    /// Send a message to a specific session.
    pub fn send_to(
        &mut self,
        session_id: &str,
        topic: impl Into<String>,
        payload: impl Into<String>,
        msg_type: MessageType,
    ) -> Result<()> {
        let topic = topic.into();
        let payload = payload.into();
        let size_bytes = payload.len();

        if size_bytes > self.config.max_message_size {
            return Err(Error::Configuration(format!(
                "Message size ({} bytes) exceeds max ({})",
                size_bytes, self.config.max_message_size
            )));
        }

        let session = self
            .sessions
            .get_mut(session_id)
            .ok_or_else(|| Error::Configuration(format!("Unknown session '{}'", session_id)))?;

        if session.state != SessionState::Open {
            return Err(Error::Configuration(format!(
                "Session '{}' is not open (state: {})",
                session_id, session.state
            )));
        }

        let msg_id = self.next_message_id;
        self.next_message_id += 1;

        let msg = WsMessage {
            id: msg_id,
            msg_type,
            topic,
            payload,
            size_bytes,
            tick: self.tick,
        };

        let was_dropped = session.enqueue(msg);
        session.last_activity_tick = self.tick;

        self.sent_this_tick += 1;
        self.bytes_this_tick += size_bytes as u64;
        self.stats.total_messages_sent += 1;
        self.stats.total_bytes_sent += size_bytes as u64;
        if was_dropped {
            self.dropped_this_tick += 1;
            self.stats.total_messages_dropped += 1;
        }

        Ok(())
    }

    // -------------------------------------------------------------------
    // Receiving messages
    // -------------------------------------------------------------------

    /// Record that a message was received from a session.
    pub fn receive_from(
        &mut self,
        session_id: &str,
        size_bytes: usize,
    ) -> Result<()> {
        let session = self
            .sessions
            .get_mut(session_id)
            .ok_or_else(|| Error::Configuration(format!("Unknown session '{}'", session_id)))?;

        session.total_received += 1;
        session.total_bytes_received += size_bytes as u64;
        session.last_activity_tick = self.tick;

        self.received_this_tick += 1;
        self.stats.total_messages_received += 1;
        self.stats.total_bytes_received += size_bytes as u64;

        Ok(())
    }

    // -------------------------------------------------------------------
    // Buffer management
    // -------------------------------------------------------------------

    /// Drain all buffered outbound messages for a session.
    pub fn drain_session_buffer(&mut self, session_id: &str) -> Result<Vec<WsMessage>> {
        let session = self
            .sessions
            .get_mut(session_id)
            .ok_or_else(|| Error::Configuration(format!("Unknown session '{}'", session_id)))?;
        Ok(session.drain_buffer())
    }

    /// Number of messages buffered for a specific session.
    pub fn session_buffered_count(&self, session_id: &str) -> usize {
        self.sessions
            .get(session_id)
            .map(|s| s.buffered_count())
            .unwrap_or(0)
    }

    /// Total messages buffered across all sessions.
    pub fn total_buffered_count(&self) -> usize {
        self.sessions.values().map(|s| s.buffered_count()).sum()
    }

    // -------------------------------------------------------------------
    // Idle detection
    // -------------------------------------------------------------------

    /// Return session IDs that have been idle longer than the configured
    /// timeout.
    pub fn idle_sessions(&self) -> Vec<&str> {
        self.sessions
            .values()
            .filter(|s| {
                s.state == SessionState::Open
                    && self.tick.saturating_sub(s.last_activity_tick)
                        >= self.config.idle_timeout_ticks
            })
            .map(|s| s.id.as_str())
            .collect()
    }

    /// Close all idle sessions and return the number closed.
    pub fn close_idle_sessions(&mut self) -> usize {
        let idle_ids: Vec<String> = self.idle_sessions().iter().map(|s| s.to_string()).collect();
        let count = idle_ids.len();
        for id in &idle_ids {
            if let Some(session) = self.sessions.get_mut(id.as_str()) {
                session.state = SessionState::Closed;
            }
            self.sessions.remove(id.as_str());
            self.stats.total_sessions_closed += 1;
            self.stats.total_timeouts += 1;
        }
        count
    }

    // -------------------------------------------------------------------
    // Tick
    // -------------------------------------------------------------------

    /// Advance the manager by one tick, updating EMA and windowed
    /// diagnostics.
    pub fn tick(&mut self) {
        self.tick += 1;

        let active = self.active_session_count();

        let snapshot = TickSnapshot {
            active_sessions: active,
            messages_sent: self.sent_this_tick,
            messages_received: self.received_this_tick,
            messages_dropped: self.dropped_this_tick,
            bytes_sent: self.bytes_this_tick,
        };

        // EMA update.
        let alpha = self.config.ema_decay;
        let sent = self.sent_this_tick as f64;
        let bytes = self.bytes_this_tick as f64;

        if !self.ema_initialized {
            self.stats.ema_active_sessions = active as f64;
            self.stats.ema_messages_per_tick = sent;
            self.stats.ema_bytes_per_tick = bytes;
            self.ema_initialized = true;
        } else {
            self.stats.ema_active_sessions =
                alpha * active as f64 + (1.0 - alpha) * self.stats.ema_active_sessions;
            self.stats.ema_messages_per_tick =
                alpha * sent + (1.0 - alpha) * self.stats.ema_messages_per_tick;
            self.stats.ema_bytes_per_tick =
                alpha * bytes + (1.0 - alpha) * self.stats.ema_bytes_per_tick;
        }

        // Window update.
        if self.recent.len() >= self.config.window_size {
            self.recent.pop_front();
        }
        self.recent.push_back(snapshot);

        // Reset per-tick counters.
        self.sent_this_tick = 0;
        self.bytes_this_tick = 0;
        self.received_this_tick = 0;
        self.dropped_this_tick = 0;
    }

    /// Current tick counter.
    pub fn current_tick(&self) -> u64 {
        self.tick
    }

    /// Alias for `tick()`.
    pub fn process(&mut self) {
        self.tick();
    }

    // -------------------------------------------------------------------
    // Statistics & diagnostics
    // -------------------------------------------------------------------

    /// Returns a reference to cumulative statistics.
    pub fn stats(&self) -> &WebsocketStats {
        &self.stats
    }

    /// Returns a reference to the configuration.
    pub fn config(&self) -> &WebsocketConfig {
        &self.config
    }

    /// EMA-smoothed active session count.
    pub fn smoothed_active_sessions(&self) -> f64 {
        self.stats.ema_active_sessions
    }

    /// EMA-smoothed messages sent per tick.
    pub fn smoothed_messages_per_tick(&self) -> f64 {
        self.stats.ema_messages_per_tick
    }

    /// EMA-smoothed bytes sent per tick.
    pub fn smoothed_bytes_per_tick(&self) -> f64 {
        self.stats.ema_bytes_per_tick
    }

    /// Windowed average active session count.
    pub fn windowed_active_sessions(&self) -> Option<f64> {
        if self.recent.is_empty() {
            return None;
        }
        let sum: f64 = self
            .recent
            .iter()
            .map(|s| s.active_sessions as f64)
            .sum();
        Some(sum / self.recent.len() as f64)
    }

    /// Windowed average messages sent per tick.
    pub fn windowed_messages_per_tick(&self) -> Option<f64> {
        if self.recent.is_empty() {
            return None;
        }
        let sum: f64 = self.recent.iter().map(|s| s.messages_sent as f64).sum();
        Some(sum / self.recent.len() as f64)
    }

    /// Windowed average messages dropped per tick.
    pub fn windowed_drops_per_tick(&self) -> Option<f64> {
        if self.recent.is_empty() {
            return None;
        }
        let sum: f64 = self
            .recent
            .iter()
            .map(|s| s.messages_dropped as f64)
            .sum();
        Some(sum / self.recent.len() as f64)
    }

    /// Whether throughput appears to be declining over the window.
    pub fn is_throughput_declining(&self) -> bool {
        if self.recent.len() < 4 {
            return false;
        }
        let mid = self.recent.len() / 2;
        let first_half: f64 = self
            .recent
            .iter()
            .take(mid)
            .map(|s| s.messages_sent as f64)
            .sum::<f64>()
            / mid as f64;
        let second_half: f64 = self
            .recent
            .iter()
            .skip(mid)
            .map(|s| s.messages_sent as f64)
            .sum::<f64>()
            / (self.recent.len() - mid) as f64;
        second_half < first_half * 0.8
    }

    // -------------------------------------------------------------------
    // Reset
    // -------------------------------------------------------------------

    /// Reset all state, closing all sessions and clearing statistics.
    pub fn reset(&mut self) {
        self.sessions.clear();
        self.next_message_id = 1;
        self.tick = 0;
        self.sent_this_tick = 0;
        self.bytes_this_tick = 0;
        self.received_this_tick = 0;
        self.dropped_this_tick = 0;
        self.ema_initialized = false;
        self.recent.clear();
        self.stats = WebsocketStats::default();
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> WebsocketConfig {
        WebsocketConfig {
            max_sessions: 4,
            session_buffer_size: 3,
            max_message_size: 1024,
            idle_timeout_ticks: 5,
            ema_decay: 0.5,
            window_size: 5,
        }
    }

    // -------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------

    #[test]
    fn test_new_default() {
        let ws = Websocket::new();
        assert_eq!(ws.session_count(), 0);
        assert_eq!(ws.current_tick(), 0);
        assert_eq!(ws.active_session_count(), 0);
    }

    #[test]
    fn test_with_config() {
        let ws = Websocket::with_config(small_config());
        assert_eq!(ws.config().max_sessions, 4);
        assert_eq!(ws.config().session_buffer_size, 3);
    }

    // -------------------------------------------------------------------
    // Session management
    // -------------------------------------------------------------------

    #[test]
    fn test_open_session() {
        let mut ws = Websocket::with_config(small_config());
        ws.open_session("s1", "client-a").unwrap();
        assert_eq!(ws.session_count(), 1);
        assert_eq!(ws.active_session_count(), 1);
        let session = ws.session("s1").unwrap();
        assert_eq!(session.state, SessionState::Open);
        assert_eq!(session.client, "client-a");
    }

    #[test]
    fn test_open_duplicate_fails() {
        let mut ws = Websocket::with_config(small_config());
        ws.open_session("s1", "c").unwrap();
        assert!(ws.open_session("s1", "c").is_err());
    }

    #[test]
    fn test_open_at_capacity() {
        let mut ws = Websocket::with_config(small_config());
        for i in 0..4 {
            ws.open_session(format!("s{}", i), "c").unwrap();
        }
        assert!(ws.open_session("overflow", "c").is_err());
    }

    #[test]
    fn test_close_session() {
        let mut ws = Websocket::with_config(small_config());
        ws.open_session("s1", "c").unwrap();
        ws.close_session("s1").unwrap();
        assert_eq!(ws.session_count(), 0);
        assert_eq!(ws.stats().total_sessions_closed, 1);
    }

    #[test]
    fn test_close_unknown_fails() {
        let mut ws = Websocket::with_config(small_config());
        assert!(ws.close_session("nope").is_err());
    }

    #[test]
    fn test_session_ids() {
        let mut ws = Websocket::with_config(small_config());
        ws.open_session("b", "c").unwrap();
        ws.open_session("a", "c").unwrap();
        let mut ids = ws.session_ids();
        ids.sort();
        assert_eq!(ids, vec!["a", "b"]);
    }

    // -------------------------------------------------------------------
    // Subscriptions
    // -------------------------------------------------------------------

    #[test]
    fn test_subscribe() {
        let mut ws = Websocket::with_config(small_config());
        ws.open_session("s1", "c").unwrap();
        ws.subscribe("s1", vec!["market".into(), "signals".into()])
            .unwrap();
        let session = ws.session("s1").unwrap();
        assert_eq!(session.subscriptions.len(), 2);
    }

    #[test]
    fn test_subscribe_no_duplicate() {
        let mut ws = Websocket::with_config(small_config());
        ws.open_session("s1", "c").unwrap();
        ws.subscribe("s1", vec!["market".into()]).unwrap();
        ws.subscribe("s1", vec!["market".into()]).unwrap();
        assert_eq!(ws.session("s1").unwrap().subscriptions.len(), 1);
    }

    #[test]
    fn test_unsubscribe() {
        let mut ws = Websocket::with_config(small_config());
        ws.open_session("s1", "c").unwrap();
        ws.subscribe("s1", vec!["market".into(), "signals".into()])
            .unwrap();
        ws.unsubscribe("s1", &["market".to_string()]).unwrap();
        assert_eq!(ws.session("s1").unwrap().subscriptions.len(), 1);
        assert_eq!(ws.session("s1").unwrap().subscriptions[0], "signals");
    }

    #[test]
    fn test_subscribe_unknown_fails() {
        let mut ws = Websocket::with_config(small_config());
        assert!(ws.subscribe("nope", vec!["t".into()]).is_err());
    }

    // -------------------------------------------------------------------
    // Sending: broadcast
    // -------------------------------------------------------------------

    #[test]
    fn test_broadcast_all() {
        let mut ws = Websocket::with_config(small_config());
        ws.open_session("s1", "c").unwrap();
        ws.open_session("s2", "c").unwrap();

        let delivered = ws
            .broadcast("market.price", "100.5", MessageType::Text)
            .unwrap();
        assert_eq!(delivered, 2);
        assert_eq!(ws.session_buffered_count("s1"), 1);
        assert_eq!(ws.session_buffered_count("s2"), 1);
        assert_eq!(ws.stats().total_messages_sent, 2);
    }

    #[test]
    fn test_broadcast_with_subscription_filter() {
        let mut ws = Websocket::with_config(small_config());
        ws.open_session("s1", "c").unwrap();
        ws.open_session("s2", "c").unwrap();
        ws.subscribe("s1", vec!["market".into()]).unwrap();
        ws.subscribe("s2", vec!["signals".into()]).unwrap();

        let delivered = ws
            .broadcast("market.price", "100", MessageType::Text)
            .unwrap();
        assert_eq!(delivered, 1); // only s1
        assert_eq!(ws.session_buffered_count("s1"), 1);
        assert_eq!(ws.session_buffered_count("s2"), 0);
    }

    #[test]
    fn test_broadcast_empty_subscription_matches_all() {
        let mut ws = Websocket::with_config(small_config());
        ws.open_session("s1", "c").unwrap(); // no subscriptions = all topics

        let delivered = ws
            .broadcast("anything", "data", MessageType::Text)
            .unwrap();
        assert_eq!(delivered, 1);
    }

    #[test]
    fn test_broadcast_oversized_fails() {
        let mut ws = Websocket::with_config(small_config());
        ws.open_session("s1", "c").unwrap();
        let big_payload = "x".repeat(2000); // > max_message_size=1024
        assert!(ws
            .broadcast("topic", &big_payload, MessageType::Text)
            .is_err());
    }

    #[test]
    fn test_broadcast_buffer_overflow_drops_oldest() {
        let mut ws = Websocket::with_config(small_config()); // buffer_size=3
        ws.open_session("s1", "c").unwrap();

        for i in 0..5 {
            ws.broadcast("t", &format!("msg{}", i), MessageType::Text)
                .unwrap();
        }

        // Buffer should have the last 3 messages.
        assert_eq!(ws.session_buffered_count("s1"), 3);
        assert_eq!(ws.session("s1").unwrap().total_dropped, 2);
        assert_eq!(ws.stats().total_messages_dropped, 2);
    }

    // -------------------------------------------------------------------
    // Sending: targeted
    // -------------------------------------------------------------------

    #[test]
    fn test_send_to() {
        let mut ws = Websocket::with_config(small_config());
        ws.open_session("s1", "c").unwrap();
        ws.send_to("s1", "topic", "data", MessageType::Text)
            .unwrap();
        assert_eq!(ws.session_buffered_count("s1"), 1);
        assert_eq!(ws.stats().total_messages_sent, 1);
    }

    #[test]
    fn test_send_to_unknown_fails() {
        let mut ws = Websocket::with_config(small_config());
        assert!(ws
            .send_to("nope", "t", "d", MessageType::Text)
            .is_err());
    }

    #[test]
    fn test_send_to_oversized_fails() {
        let mut ws = Websocket::with_config(small_config());
        ws.open_session("s1", "c").unwrap();
        let big = "x".repeat(2000);
        assert!(ws.send_to("s1", "t", &big, MessageType::Text).is_err());
    }

    // -------------------------------------------------------------------
    // Receiving
    // -------------------------------------------------------------------

    #[test]
    fn test_receive_from() {
        let mut ws = Websocket::with_config(small_config());
        ws.open_session("s1", "c").unwrap();
        ws.receive_from("s1", 128).unwrap();
        assert_eq!(ws.session("s1").unwrap().total_received, 1);
        assert_eq!(ws.session("s1").unwrap().total_bytes_received, 128);
        assert_eq!(ws.stats().total_messages_received, 1);
        assert_eq!(ws.stats().total_bytes_received, 128);
    }

    #[test]
    fn test_receive_from_unknown_fails() {
        let mut ws = Websocket::with_config(small_config());
        assert!(ws.receive_from("nope", 10).is_err());
    }

    // -------------------------------------------------------------------
    // Buffer management
    // -------------------------------------------------------------------

    #[test]
    fn test_drain_session_buffer() {
        let mut ws = Websocket::with_config(small_config());
        ws.open_session("s1", "c").unwrap();
        ws.send_to("s1", "t", "m1", MessageType::Text).unwrap();
        ws.send_to("s1", "t", "m2", MessageType::Text).unwrap();

        let msgs = ws.drain_session_buffer("s1").unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(ws.session_buffered_count("s1"), 0);
    }

    #[test]
    fn test_drain_unknown_fails() {
        let mut ws = Websocket::with_config(small_config());
        assert!(ws.drain_session_buffer("nope").is_err());
    }

    #[test]
    fn test_total_buffered_count() {
        let mut ws = Websocket::with_config(small_config());
        ws.open_session("s1", "c").unwrap();
        ws.open_session("s2", "c").unwrap();
        ws.send_to("s1", "t", "m", MessageType::Text).unwrap();
        ws.send_to("s2", "t", "m", MessageType::Text).unwrap();
        ws.send_to("s2", "t", "m", MessageType::Text).unwrap();
        assert_eq!(ws.total_buffered_count(), 3);
    }

    // -------------------------------------------------------------------
    // Idle detection
    // -------------------------------------------------------------------

    #[test]
    fn test_idle_sessions() {
        let mut ws = Websocket::with_config(small_config()); // idle_timeout=5
        ws.open_session("s1", "c").unwrap();
        ws.open_session("s2", "c").unwrap();

        for _ in 0..5 {
            ws.tick();
        }

        let idle = ws.idle_sessions();
        assert_eq!(idle.len(), 2);
    }

    #[test]
    fn test_activity_resets_idle() {
        let mut ws = Websocket::with_config(small_config());
        ws.open_session("s1", "c").unwrap();

        for _ in 0..3 {
            ws.tick();
        }

        // Activity on s1 resets the idle timer.
        ws.receive_from("s1", 10).unwrap();

        for _ in 0..3 {
            ws.tick();
        }

        // s1 should not be idle yet (only 3 ticks since activity, need 5).
        let idle = ws.idle_sessions();
        assert!(idle.is_empty());
    }

    #[test]
    fn test_close_idle_sessions() {
        let mut ws = Websocket::with_config(small_config());
        ws.open_session("s1", "c").unwrap();

        for _ in 0..6 {
            ws.tick();
        }

        let closed = ws.close_idle_sessions();
        assert_eq!(closed, 1);
        assert_eq!(ws.session_count(), 0);
        assert_eq!(ws.stats().total_timeouts, 1);
    }

    // -------------------------------------------------------------------
    // Tick & EMA
    // -------------------------------------------------------------------

    #[test]
    fn test_tick_increments() {
        let mut ws = Websocket::with_config(small_config());
        ws.tick();
        ws.tick();
        assert_eq!(ws.current_tick(), 2);
    }

    #[test]
    fn test_process_alias() {
        let mut ws = Websocket::with_config(small_config());
        ws.process();
        assert_eq!(ws.current_tick(), 1);
    }

    #[test]
    fn test_ema_initialises_on_first_tick() {
        let mut ws = Websocket::with_config(small_config());
        ws.open_session("s1", "c").unwrap();
        ws.send_to("s1", "t", "hello", MessageType::Text).unwrap();
        ws.tick();
        assert!((ws.smoothed_active_sessions() - 1.0).abs() < 1e-9);
        assert!((ws.smoothed_messages_per_tick() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_ema_blends_on_subsequent_ticks() {
        let mut ws = Websocket::with_config(small_config()); // ema_decay=0.5
        ws.open_session("s1", "c").unwrap();
        ws.send_to("s1", "t", "a", MessageType::Text).unwrap();
        ws.tick(); // sent=1, ema=1.0
        // No sends this tick.
        ws.tick(); // sent=0, ema = 0.5*0 + 0.5*1.0 = 0.5
        assert!((ws.smoothed_messages_per_tick() - 0.5).abs() < 1e-9);
    }

    // -------------------------------------------------------------------
    // Windowed diagnostics
    // -------------------------------------------------------------------

    #[test]
    fn test_windowed_active_sessions_empty() {
        let ws = Websocket::with_config(small_config());
        assert!(ws.windowed_active_sessions().is_none());
    }

    #[test]
    fn test_windowed_active_sessions() {
        let mut ws = Websocket::with_config(small_config());
        ws.open_session("s1", "c").unwrap();
        ws.tick();
        ws.tick();
        let avg = ws.windowed_active_sessions().unwrap();
        assert!((avg - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_windowed_messages_per_tick() {
        let mut ws = Websocket::with_config(small_config());
        ws.open_session("s1", "c").unwrap();
        ws.send_to("s1", "t", "m", MessageType::Text).unwrap();
        ws.tick(); // sent=1
        ws.tick(); // sent=0
        let avg = ws.windowed_messages_per_tick().unwrap();
        assert!((avg - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_windowed_drops_per_tick() {
        let mut ws = Websocket::with_config(small_config());
        ws.open_session("s1", "c").unwrap();
        ws.tick(); // no drops
        let avg = ws.windowed_drops_per_tick().unwrap();
        assert!((avg - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_is_throughput_declining_insufficient() {
        let mut ws = Websocket::with_config(small_config());
        ws.tick();
        assert!(!ws.is_throughput_declining());
    }

    #[test]
    fn test_is_throughput_declining() {
        let mut ws = Websocket::with_config(small_config());
        ws.open_session("s1", "c").unwrap();

        // First two ticks: send messages.
        ws.send_to("s1", "t", "m", MessageType::Text).unwrap();
        ws.send_to("s1", "t", "m", MessageType::Text).unwrap();
        ws.tick();

        ws.send_to("s1", "t", "m", MessageType::Text).unwrap();
        ws.tick();

        // Next two ticks: no messages.
        ws.tick();
        ws.tick();

        assert!(ws.is_throughput_declining());
    }

    #[test]
    fn test_window_rolls() {
        let mut ws = Websocket::with_config(small_config()); // window_size=5
        for _ in 0..20 {
            ws.tick();
        }
        assert!(ws.recent.len() <= 5);
    }

    // -------------------------------------------------------------------
    // Per-session stats
    // -------------------------------------------------------------------

    #[test]
    fn test_session_total_sent() {
        let mut ws = Websocket::with_config(small_config());
        ws.open_session("s1", "c").unwrap();
        ws.send_to("s1", "t", "m1", MessageType::Text).unwrap();
        ws.send_to("s1", "t", "m2", MessageType::Text).unwrap();
        assert_eq!(ws.session("s1").unwrap().total_sent, 2);
    }

    #[test]
    fn test_session_bytes_sent() {
        let mut ws = Websocket::with_config(small_config());
        ws.open_session("s1", "c").unwrap();
        ws.send_to("s1", "t", "hello", MessageType::Text).unwrap(); // 5 bytes
        assert_eq!(ws.session("s1").unwrap().total_bytes_sent, 5);
    }

    #[test]
    fn test_session_last_activity_tick() {
        let mut ws = Websocket::with_config(small_config());
        ws.open_session("s1", "c").unwrap();
        ws.tick();
        ws.tick();
        ws.send_to("s1", "t", "m", MessageType::Text).unwrap();
        assert_eq!(ws.session("s1").unwrap().last_activity_tick, 2);
    }

    // -------------------------------------------------------------------
    // Message types
    // -------------------------------------------------------------------

    #[test]
    fn test_message_types() {
        let mut ws = Websocket::with_config(small_config());
        ws.open_session("s1", "c").unwrap();

        ws.send_to("s1", "t", "", MessageType::Ping).unwrap();
        ws.send_to("s1", "t", "{}", MessageType::Binary).unwrap();

        let msgs = ws.drain_session_buffer("s1").unwrap();
        assert_eq!(msgs[0].msg_type, MessageType::Ping);
        assert_eq!(msgs[1].msg_type, MessageType::Binary);
    }

    #[test]
    fn test_message_type_display() {
        assert_eq!(format!("{}", MessageType::Text), "Text");
        assert_eq!(format!("{}", MessageType::Binary), "Binary");
        assert_eq!(format!("{}", MessageType::Ping), "Ping");
        assert_eq!(format!("{}", MessageType::Pong), "Pong");
        assert_eq!(format!("{}", MessageType::Close), "Close");
    }

    // -------------------------------------------------------------------
    // Reset
    // -------------------------------------------------------------------

    #[test]
    fn test_reset() {
        let mut ws = Websocket::with_config(small_config());
        ws.open_session("s1", "c").unwrap();
        ws.send_to("s1", "t", "m", MessageType::Text).unwrap();
        ws.tick();

        ws.reset();

        assert_eq!(ws.session_count(), 0);
        assert_eq!(ws.current_tick(), 0);
        assert_eq!(ws.stats().total_messages_sent, 0);
        assert_eq!(ws.stats().total_sessions_opened, 0);
        assert!(ws.windowed_active_sessions().is_none());
    }

    // -------------------------------------------------------------------
    // Full lifecycle
    // -------------------------------------------------------------------

    #[test]
    fn test_full_lifecycle() {
        let mut ws = Websocket::with_config(small_config());

        // Open sessions.
        ws.open_session("client-1", "192.168.1.1").unwrap();
        ws.open_session("client-2", "192.168.1.2").unwrap();

        // Subscribe.
        ws.subscribe("client-1", vec!["market".into()]).unwrap();
        ws.subscribe("client-2", vec!["signals".into()]).unwrap();

        // Broadcast market data — only client-1 should receive.
        let delivered = ws
            .broadcast("market.btc", "50000", MessageType::Text)
            .unwrap();
        assert_eq!(delivered, 1);
        ws.tick();

        // Send a signal to client-2 directly.
        ws.send_to("client-2", "signals.buy", "BTC", MessageType::Text)
            .unwrap();
        ws.tick();

        // Receive from client-1.
        ws.receive_from("client-1", 64).unwrap();
        ws.tick();

        // Drain client-1 buffer.
        let msgs = ws.drain_session_buffer("client-1").unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].topic, "market.btc");

        // Drain client-2 buffer.
        let msgs = ws.drain_session_buffer("client-2").unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].topic, "signals.buy");

        // Close client-1.
        ws.close_session("client-1").unwrap();
        assert_eq!(ws.active_session_count(), 1);

        // Stats.
        assert_eq!(ws.stats().total_sessions_opened, 2);
        assert_eq!(ws.stats().total_sessions_closed, 1);
        assert_eq!(ws.stats().total_messages_sent, 2);
        assert_eq!(ws.stats().total_messages_received, 1);
        assert!(ws.smoothed_active_sessions() > 0.0);
        assert!(ws.windowed_active_sessions().is_some());
    }

    #[test]
    fn test_session_state_display() {
        assert_eq!(format!("{}", SessionState::Connecting), "Connecting");
        assert_eq!(format!("{}", SessionState::Open), "Open");
        assert_eq!(format!("{}", SessionState::Closing), "Closing");
        assert_eq!(format!("{}", SessionState::Closed), "Closed");
    }

    // -------------------------------------------------------------------
    // Message ID monotonicity
    // -------------------------------------------------------------------

    #[test]
    fn test_message_ids_monotonic() {
        let mut ws = Websocket::with_config(small_config());
        ws.open_session("s1", "c").unwrap();
        ws.send_to("s1", "t", "a", MessageType::Text).unwrap();
        ws.send_to("s1", "t", "b", MessageType::Text).unwrap();
        ws.send_to("s1", "t", "c", MessageType::Text).unwrap();
        let msgs = ws.drain_session_buffer("s1").unwrap();
        assert!(msgs[0].id < msgs[1].id);
        assert!(msgs[1].id < msgs[2].id);
    }
}
