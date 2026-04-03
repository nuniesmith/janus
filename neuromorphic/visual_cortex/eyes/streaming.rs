//! Real-time data streaming
//!
//! Part of the Visual Cortex region
//! Component: eyes
//!
//! Implements real-time market data streaming with:
//! - Buffered stream processing
//! - Rate limiting and backpressure handling
//! - Multiple data source support
//! - Automatic reconnection
//! - Data quality monitoring
//! - Stream multiplexing

use crate::common::{Error, Result};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// Stream status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamStatus {
    /// Stream is connected and receiving data
    Connected,
    /// Stream is connecting
    Connecting,
    /// Stream is disconnected
    Disconnected,
    /// Stream is paused (backpressure)
    Paused,
    /// Stream has encountered an error
    Error,
    /// Stream is reconnecting
    Reconnecting,
}

/// Type of streaming data
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StreamType {
    /// Trade data
    Trades,
    /// Order book updates
    OrderBook,
    /// OHLCV candles
    Candles,
    /// Ticker updates
    Ticker,
    /// Liquidations
    Liquidations,
    /// Funding rates
    FundingRates,
    /// Custom stream
    Custom,
}

/// Configuration for streaming
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Maximum buffer size per stream
    pub max_buffer_size: usize,
    /// Rate limit (messages per second)
    pub rate_limit: f64,
    /// Reconnect delay in milliseconds
    pub reconnect_delay_ms: u64,
    /// Maximum reconnect attempts
    pub max_reconnect_attempts: u32,
    /// Backpressure threshold (buffer fill ratio)
    pub backpressure_threshold: f64,
    /// Enable data quality monitoring
    pub quality_monitoring: bool,
    /// Stale data threshold in milliseconds
    pub stale_threshold_ms: u64,
    /// Batch size for processing
    pub batch_size: usize,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            max_buffer_size: 10000,
            rate_limit: 1000.0,
            reconnect_delay_ms: 1000,
            max_reconnect_attempts: 10,
            backpressure_threshold: 0.8,
            quality_monitoring: true,
            stale_threshold_ms: 5000,
            batch_size: 100,
        }
    }
}

/// A single data point from a stream
#[derive(Debug, Clone)]
pub struct StreamData {
    /// Stream identifier
    pub stream_id: String,
    /// Data type
    pub stream_type: StreamType,
    /// Symbol/instrument
    pub symbol: String,
    /// Timestamp in milliseconds
    pub timestamp: i64,
    /// Sequence number (for ordering)
    pub sequence: u64,
    /// Price (if applicable)
    pub price: Option<f64>,
    /// Volume/size (if applicable)
    pub volume: Option<f64>,
    /// Side (buy/sell, 1/-1)
    pub side: Option<i8>,
    /// Additional data as key-value pairs
    pub extra: HashMap<String, f64>,
    /// Reception timestamp
    pub received_at: Instant,
}

impl StreamData {
    /// Create a new stream data point
    pub fn new(stream_id: &str, stream_type: StreamType, symbol: &str, timestamp: i64) -> Self {
        Self {
            stream_id: stream_id.to_string(),
            stream_type,
            symbol: symbol.to_string(),
            timestamp,
            sequence: 0,
            price: None,
            volume: None,
            side: None,
            extra: HashMap::new(),
            received_at: Instant::now(),
        }
    }

    /// Check if data is stale
    pub fn is_stale(&self, threshold_ms: u64) -> bool {
        self.received_at.elapsed() > Duration::from_millis(threshold_ms)
    }

    /// Get latency from exchange timestamp to reception
    pub fn latency_ms(&self) -> u64 {
        let now_ms = chrono::Utc::now().timestamp_millis();
        (now_ms - self.timestamp).max(0) as u64
    }
}

/// Stream statistics
#[derive(Debug, Clone, Default)]
pub struct StreamStats {
    /// Total messages received
    pub messages_received: u64,
    /// Total messages processed
    pub messages_processed: u64,
    /// Messages dropped (buffer overflow)
    pub messages_dropped: u64,
    /// Current buffer size
    pub buffer_size: usize,
    /// Average latency in milliseconds
    pub avg_latency_ms: f64,
    /// Maximum latency observed
    pub max_latency_ms: u64,
    /// Messages per second (current rate)
    pub messages_per_second: f64,
    /// Reconnection count
    pub reconnect_count: u32,
    /// Last message timestamp
    pub last_message_time: Option<Instant>,
    /// Gaps detected (missing sequences)
    pub gaps_detected: u64,
}

/// Individual stream state
#[derive(Debug)]
struct StreamState {
    /// Stream identifier
    id: String,
    /// Stream type
    stream_type: StreamType,
    /// Symbols subscribed
    symbols: Vec<String>,
    /// Current status
    status: StreamStatus,
    /// Data buffer
    buffer: VecDeque<StreamData>,
    /// Statistics
    stats: StreamStats,
    /// Last sequence number seen
    last_sequence: u64,
    /// Reconnect attempts
    reconnect_attempts: u32,
    /// Last activity time
    last_activity: Instant,
    /// Latency sum for averaging
    latency_sum: u64,
    /// Message count for rate calculation
    rate_window_count: u64,
    /// Rate window start
    rate_window_start: Instant,
}

impl StreamState {
    fn new(id: &str, stream_type: StreamType) -> Self {
        Self {
            id: id.to_string(),
            stream_type,
            symbols: Vec::new(),
            status: StreamStatus::Disconnected,
            buffer: VecDeque::new(),
            stats: StreamStats::default(),
            last_sequence: 0,
            reconnect_attempts: 0,
            last_activity: Instant::now(),
            latency_sum: 0,
            rate_window_count: 0,
            rate_window_start: Instant::now(),
        }
    }

    fn update_rate(&mut self) {
        let elapsed = self.rate_window_start.elapsed().as_secs_f64();
        if elapsed >= 1.0 {
            self.stats.messages_per_second = self.rate_window_count as f64 / elapsed;
            self.rate_window_count = 0;
            self.rate_window_start = Instant::now();
        }
    }
}

/// Quality metrics for stream data
#[derive(Debug, Clone, Default)]
pub struct StreamQuality {
    /// Data freshness score (0.0 - 1.0)
    pub freshness: f64,
    /// Completeness score (no gaps)
    pub completeness: f64,
    /// Latency score
    pub latency_score: f64,
    /// Overall quality score
    pub overall: f64,
    /// Number of stale data points
    pub stale_count: u64,
}

/// Real-time data streaming manager
///
/// Manages multiple data streams with buffering, rate limiting,
/// and quality monitoring.
pub struct Streaming {
    /// Configuration
    config: StreamingConfig,
    /// Active streams
    streams: HashMap<String, StreamState>,
    /// Global statistics
    global_stats: StreamStats,
    /// Quality metrics
    quality: StreamQuality,
    /// Paused streams (backpressure)
    paused_streams: Vec<String>,
    /// Stream priority order
    priority_order: Vec<String>,
}

impl Default for Streaming {
    fn default() -> Self {
        Self::new()
    }
}

impl Streaming {
    /// Create a new streaming manager with default config
    pub fn new() -> Self {
        Self::with_config(StreamingConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: StreamingConfig) -> Self {
        Self {
            config,
            streams: HashMap::new(),
            global_stats: StreamStats::default(),
            quality: StreamQuality::default(),
            paused_streams: Vec::new(),
            priority_order: Vec::new(),
        }
    }

    /// Main processing function
    pub fn process(&mut self) -> Result<Vec<StreamData>> {
        self.update_quality();
        self.check_backpressure();
        self.drain_buffers()
    }

    /// Register a new stream
    pub fn register_stream(
        &mut self,
        stream_id: &str,
        stream_type: StreamType,
        symbols: Vec<String>,
    ) -> Result<()> {
        if self.streams.contains_key(stream_id) {
            return Err(Error::InvalidInput(format!(
                "Stream {} already registered",
                stream_id
            )));
        }

        let mut state = StreamState::new(stream_id, stream_type);
        state.symbols = symbols;

        self.streams.insert(stream_id.to_string(), state);
        self.priority_order.push(stream_id.to_string());

        Ok(())
    }

    /// Unregister a stream
    pub fn unregister_stream(&mut self, stream_id: &str) -> bool {
        self.priority_order.retain(|id| id != stream_id);
        self.streams.remove(stream_id).is_some()
    }

    /// Connect a stream (mark as connecting)
    pub fn connect(&mut self, stream_id: &str) -> Result<()> {
        let stream = self
            .streams
            .get_mut(stream_id)
            .ok_or_else(|| Error::InvalidInput(format!("Stream {} not found", stream_id)))?;

        stream.status = StreamStatus::Connecting;
        stream.reconnect_attempts = 0;

        Ok(())
    }

    /// Mark stream as connected
    pub fn on_connected(&mut self, stream_id: &str) -> Result<()> {
        let stream = self
            .streams
            .get_mut(stream_id)
            .ok_or_else(|| Error::InvalidInput(format!("Stream {} not found", stream_id)))?;

        stream.status = StreamStatus::Connected;
        stream.last_activity = Instant::now();

        Ok(())
    }

    /// Mark stream as disconnected
    pub fn on_disconnected(&mut self, stream_id: &str) {
        if let Some(stream) = self.streams.get_mut(stream_id) {
            stream.status = StreamStatus::Disconnected;
            stream.stats.reconnect_count += 1;
        }
    }

    /// Handle incoming data
    pub fn on_data(&mut self, data: StreamData) -> Result<bool> {
        let stream = self
            .streams
            .get_mut(&data.stream_id)
            .ok_or_else(|| Error::InvalidInput(format!("Stream {} not found", data.stream_id)))?;

        // Check if stream is paused
        if stream.status == StreamStatus::Paused {
            stream.stats.messages_dropped += 1;
            self.global_stats.messages_dropped += 1;
            return Ok(false);
        }

        // Check buffer overflow
        if stream.buffer.len() >= self.config.max_buffer_size {
            stream.stats.messages_dropped += 1;
            self.global_stats.messages_dropped += 1;
            return Ok(false);
        }

        // Check for sequence gaps
        if data.sequence > 0 && stream.last_sequence > 0 {
            if data.sequence > stream.last_sequence + 1 {
                let gap = data.sequence - stream.last_sequence - 1;
                stream.stats.gaps_detected += gap;
                self.global_stats.gaps_detected += gap;
            }
        }
        stream.last_sequence = data.sequence;

        // Update latency stats
        let latency = data.latency_ms();
        stream.latency_sum += latency;
        stream.stats.messages_received += 1;
        if latency > stream.stats.max_latency_ms {
            stream.stats.max_latency_ms = latency;
        }
        stream.stats.avg_latency_ms =
            stream.latency_sum as f64 / stream.stats.messages_received as f64;

        // Update rate
        stream.rate_window_count += 1;
        stream.update_rate();

        // Update activity
        stream.last_activity = Instant::now();
        stream.stats.last_message_time = Some(Instant::now());
        stream.stats.buffer_size = stream.buffer.len() + 1;

        // Add to buffer
        stream.buffer.push_back(data);

        // Update global stats
        self.global_stats.messages_received += 1;

        Ok(true)
    }

    /// Drain data from buffers (batch processing)
    pub fn drain_buffers(&mut self) -> Result<Vec<StreamData>> {
        let mut batch = Vec::with_capacity(self.config.batch_size);

        // Process streams in priority order
        for stream_id in &self.priority_order.clone() {
            if batch.len() >= self.config.batch_size {
                break;
            }

            if let Some(stream) = self.streams.get_mut(stream_id) {
                let to_take = (self.config.batch_size - batch.len()).min(stream.buffer.len());

                for _ in 0..to_take {
                    if let Some(data) = stream.buffer.pop_front() {
                        batch.push(data);
                        stream.stats.messages_processed += 1;
                        self.global_stats.messages_processed += 1;
                    }
                }

                stream.stats.buffer_size = stream.buffer.len();
            }
        }

        Ok(batch)
    }

    /// Get data for a specific stream
    pub fn drain_stream(&mut self, stream_id: &str, max_count: usize) -> Vec<StreamData> {
        let mut result = Vec::new();

        if let Some(stream) = self.streams.get_mut(stream_id) {
            let to_take = max_count.min(stream.buffer.len());

            for _ in 0..to_take {
                if let Some(data) = stream.buffer.pop_front() {
                    result.push(data);
                    stream.stats.messages_processed += 1;
                    self.global_stats.messages_processed += 1;
                }
            }

            stream.stats.buffer_size = stream.buffer.len();
        }

        result
    }

    /// Check and apply backpressure
    fn check_backpressure(&mut self) {
        self.paused_streams.clear();

        for (id, stream) in &mut self.streams {
            let fill_ratio = stream.buffer.len() as f64 / self.config.max_buffer_size as f64;

            if fill_ratio >= self.config.backpressure_threshold {
                if stream.status == StreamStatus::Connected {
                    stream.status = StreamStatus::Paused;
                    self.paused_streams.push(id.clone());
                }
            } else if stream.status == StreamStatus::Paused {
                stream.status = StreamStatus::Connected;
            }
        }
    }

    /// Update quality metrics
    fn update_quality(&mut self) {
        if !self.config.quality_monitoring {
            return;
        }

        let mut total_freshness = 0.0;
        let mut total_completeness = 0.0;
        let mut total_latency_score = 0.0;
        let mut stale_count = 0u64;
        let active_streams = self
            .streams
            .values()
            .filter(|s| s.status == StreamStatus::Connected)
            .count();

        if active_streams == 0 {
            self.quality = StreamQuality::default();
            return;
        }

        for stream in self.streams.values() {
            if stream.status != StreamStatus::Connected {
                continue;
            }

            // Freshness: based on time since last message
            let freshness = if let Some(last_time) = stream.stats.last_message_time {
                let elapsed = last_time.elapsed().as_millis() as f64;
                (1.0 - elapsed / self.config.stale_threshold_ms as f64).max(0.0)
            } else {
                0.0
            };
            total_freshness += freshness;

            // Completeness: based on gaps
            let completeness = if stream.stats.messages_received > 0 {
                let gap_ratio = stream.stats.gaps_detected as f64
                    / (stream.stats.messages_received as f64 + stream.stats.gaps_detected as f64);
                1.0 - gap_ratio
            } else {
                1.0
            };
            total_completeness += completeness;

            // Latency score (target < 100ms)
            let latency_score = (1.0 - stream.stats.avg_latency_ms / 500.0).clamp(0.0, 1.0);
            total_latency_score += latency_score;

            // Count stale data in buffer
            stale_count += stream
                .buffer
                .iter()
                .filter(|d| d.is_stale(self.config.stale_threshold_ms))
                .count() as u64;
        }

        let n = active_streams as f64;
        self.quality = StreamQuality {
            freshness: total_freshness / n,
            completeness: total_completeness / n,
            latency_score: total_latency_score / n,
            overall: (total_freshness + total_completeness + total_latency_score) / (3.0 * n),
            stale_count,
        };
    }

    /// Get stream status
    pub fn stream_status(&self, stream_id: &str) -> Option<StreamStatus> {
        self.streams.get(stream_id).map(|s| s.status)
    }

    /// Get stream statistics
    pub fn stream_stats(&self, stream_id: &str) -> Option<&StreamStats> {
        self.streams.get(stream_id).map(|s| &s.stats)
    }

    /// Get global statistics
    pub fn global_stats(&self) -> &StreamStats {
        &self.global_stats
    }

    /// Get quality metrics
    pub fn quality(&self) -> &StreamQuality {
        &self.quality
    }

    /// Get list of active streams
    pub fn active_streams(&self) -> Vec<&str> {
        self.streams
            .iter()
            .filter(|(_, s)| s.status == StreamStatus::Connected)
            .map(|(id, _)| id.as_str())
            .collect()
    }

    /// Get list of paused streams
    pub fn paused_streams(&self) -> &[String] {
        &self.paused_streams
    }

    /// Check if any stream needs reconnection
    pub fn streams_needing_reconnect(&self) -> Vec<&str> {
        self.streams
            .iter()
            .filter(|(_, s)| {
                s.status == StreamStatus::Disconnected
                    && s.reconnect_attempts < self.config.max_reconnect_attempts
            })
            .map(|(id, _)| id.as_str())
            .collect()
    }

    /// Mark stream as attempting reconnect
    pub fn on_reconnect_attempt(&mut self, stream_id: &str) -> bool {
        if let Some(stream) = self.streams.get_mut(stream_id) {
            if stream.reconnect_attempts < self.config.max_reconnect_attempts {
                stream.status = StreamStatus::Reconnecting;
                stream.reconnect_attempts += 1;
                return true;
            }
        }
        false
    }

    /// Set stream priority (lower index = higher priority)
    pub fn set_priority(&mut self, stream_id: &str, priority: usize) {
        self.priority_order.retain(|id| id != stream_id);
        let index = priority.min(self.priority_order.len());
        self.priority_order.insert(index, stream_id.to_string());
    }

    /// Get total buffer size across all streams
    pub fn total_buffer_size(&self) -> usize {
        self.streams.values().map(|s| s.buffer.len()).sum()
    }

    /// Clear all buffers
    pub fn clear_buffers(&mut self) {
        for stream in self.streams.values_mut() {
            stream.buffer.clear();
            stream.stats.buffer_size = 0;
        }
    }

    /// Reset all statistics
    pub fn reset_stats(&mut self) {
        self.global_stats = StreamStats::default();
        for stream in self.streams.values_mut() {
            stream.stats = StreamStats::default();
            stream.latency_sum = 0;
        }
    }

    /// Reset everything
    pub fn reset(&mut self) {
        self.streams.clear();
        self.global_stats = StreamStats::default();
        self.quality = StreamQuality::default();
        self.paused_streams.clear();
        self.priority_order.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_data(stream_id: &str, sequence: u64) -> StreamData {
        let mut data = StreamData::new(
            stream_id,
            StreamType::Trades,
            "BTC-USD",
            chrono::Utc::now().timestamp_millis(),
        );
        data.sequence = sequence;
        data.price = Some(50000.0);
        data.volume = Some(1.5);
        data
    }

    #[test]
    fn test_basic() {
        let mut instance = Streaming::new();
        assert!(instance.process().is_ok());
    }

    #[test]
    fn test_register_stream() {
        let mut streaming = Streaming::new();

        streaming
            .register_stream(
                "test_stream",
                StreamType::Trades,
                vec!["BTC-USD".to_string()],
            )
            .unwrap();

        assert!(streaming.streams.contains_key("test_stream"));
        assert_eq!(
            streaming.stream_status("test_stream"),
            Some(StreamStatus::Disconnected)
        );
    }

    #[test]
    fn test_duplicate_registration() {
        let mut streaming = Streaming::new();

        streaming
            .register_stream("test", StreamType::Trades, vec![])
            .unwrap();

        let result = streaming.register_stream("test", StreamType::Trades, vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_connect_disconnect() {
        let mut streaming = Streaming::new();

        streaming
            .register_stream("test", StreamType::Trades, vec![])
            .unwrap();

        streaming.connect("test").unwrap();
        assert_eq!(
            streaming.stream_status("test"),
            Some(StreamStatus::Connecting)
        );

        streaming.on_connected("test").unwrap();
        assert_eq!(
            streaming.stream_status("test"),
            Some(StreamStatus::Connected)
        );

        streaming.on_disconnected("test");
        assert_eq!(
            streaming.stream_status("test"),
            Some(StreamStatus::Disconnected)
        );
    }

    #[test]
    fn test_data_flow() {
        let mut streaming = Streaming::new();

        streaming
            .register_stream("test", StreamType::Trades, vec!["BTC-USD".to_string()])
            .unwrap();
        streaming.on_connected("test").unwrap();

        let data = create_test_data("test", 1);
        assert!(streaming.on_data(data).unwrap());

        assert_eq!(streaming.global_stats.messages_received, 1);

        let batch = streaming.drain_buffers().unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(streaming.global_stats.messages_processed, 1);
    }

    #[test]
    fn test_sequence_gap_detection() {
        let mut streaming = Streaming::new();

        streaming
            .register_stream("test", StreamType::Trades, vec![])
            .unwrap();
        streaming.on_connected("test").unwrap();

        streaming.on_data(create_test_data("test", 1)).unwrap();
        streaming.on_data(create_test_data("test", 5)).unwrap(); // Gap of 3

        let stats = streaming.stream_stats("test").unwrap();
        assert_eq!(stats.gaps_detected, 3);
    }

    #[test]
    fn test_buffer_overflow() {
        let config = StreamingConfig {
            max_buffer_size: 5,
            ..Default::default()
        };
        let mut streaming = Streaming::with_config(config);

        streaming
            .register_stream("test", StreamType::Trades, vec![])
            .unwrap();
        streaming.on_connected("test").unwrap();

        // Fill buffer
        for i in 0..5 {
            streaming.on_data(create_test_data("test", i)).unwrap();
        }

        // This should be dropped
        let result = streaming.on_data(create_test_data("test", 5)).unwrap();
        assert!(!result);

        assert_eq!(streaming.stream_stats("test").unwrap().messages_dropped, 1);
    }

    #[test]
    fn test_backpressure() {
        let config = StreamingConfig {
            max_buffer_size: 10,
            backpressure_threshold: 0.5,
            ..Default::default()
        };
        let mut streaming = Streaming::with_config(config);

        streaming
            .register_stream("test", StreamType::Trades, vec![])
            .unwrap();
        streaming.on_connected("test").unwrap();

        // Fill to backpressure threshold
        for i in 0..6 {
            streaming.on_data(create_test_data("test", i)).unwrap();
        }

        // Trigger backpressure check
        streaming.check_backpressure();

        assert_eq!(streaming.stream_status("test"), Some(StreamStatus::Paused));
        assert!(streaming.paused_streams().contains(&"test".to_string()));
    }

    #[test]
    fn test_drain_stream() {
        let mut streaming = Streaming::new();

        streaming
            .register_stream("test", StreamType::Trades, vec![])
            .unwrap();
        streaming.on_connected("test").unwrap();

        for i in 0..10 {
            streaming.on_data(create_test_data("test", i)).unwrap();
        }

        let batch = streaming.drain_stream("test", 5);
        assert_eq!(batch.len(), 5);
        assert_eq!(streaming.stream_stats("test").unwrap().buffer_size, 5);
    }

    #[test]
    fn test_priority() {
        let mut streaming = Streaming::new();

        streaming
            .register_stream("low", StreamType::Trades, vec![])
            .unwrap();
        streaming
            .register_stream("high", StreamType::Trades, vec![])
            .unwrap();

        // High priority first
        streaming.set_priority("high", 0);

        assert_eq!(streaming.priority_order[0], "high");
    }

    #[test]
    fn test_reconnect_tracking() {
        let config = StreamingConfig {
            max_reconnect_attempts: 3,
            ..Default::default()
        };
        let mut streaming = Streaming::with_config(config);

        streaming
            .register_stream("test", StreamType::Trades, vec![])
            .unwrap();
        streaming.on_connected("test").unwrap();
        streaming.on_disconnected("test");

        // Should be able to reconnect
        let needs_reconnect = streaming.streams_needing_reconnect();
        assert!(needs_reconnect.contains(&"test"));

        // Attempt reconnects up to limit
        assert!(streaming.on_reconnect_attempt("test"));
        assert!(streaming.on_reconnect_attempt("test"));
        assert!(streaming.on_reconnect_attempt("test"));
        assert!(!streaming.on_reconnect_attempt("test")); // Exceeded limit
    }

    #[test]
    fn test_unregister() {
        let mut streaming = Streaming::new();

        streaming
            .register_stream("test", StreamType::Trades, vec![])
            .unwrap();

        assert!(streaming.unregister_stream("test"));
        assert!(!streaming.streams.contains_key("test"));
        assert!(!streaming.priority_order.contains(&"test".to_string()));
    }

    #[test]
    fn test_active_streams() {
        let mut streaming = Streaming::new();

        streaming
            .register_stream("s1", StreamType::Trades, vec![])
            .unwrap();
        streaming
            .register_stream("s2", StreamType::Trades, vec![])
            .unwrap();

        streaming.on_connected("s1").unwrap();

        let active = streaming.active_streams();
        assert_eq!(active.len(), 1);
        assert!(active.contains(&"s1"));
    }

    #[test]
    fn test_quality_metrics() {
        let mut streaming = Streaming::new();

        streaming
            .register_stream("test", StreamType::Trades, vec![])
            .unwrap();
        streaming.on_connected("test").unwrap();

        streaming.on_data(create_test_data("test", 1)).unwrap();

        streaming.update_quality();

        assert!(streaming.quality().freshness > 0.0);
        assert!(streaming.quality().overall > 0.0);
    }

    #[test]
    fn test_reset() {
        let mut streaming = Streaming::new();

        streaming
            .register_stream("test", StreamType::Trades, vec![])
            .unwrap();
        streaming.on_connected("test").unwrap();
        streaming.on_data(create_test_data("test", 1)).unwrap();

        streaming.reset();

        assert!(streaming.streams.is_empty());
        assert_eq!(streaming.global_stats.messages_received, 0);
    }

    #[test]
    fn test_clear_buffers() {
        let mut streaming = Streaming::new();

        streaming
            .register_stream("test", StreamType::Trades, vec![])
            .unwrap();
        streaming.on_connected("test").unwrap();

        for i in 0..5 {
            streaming.on_data(create_test_data("test", i)).unwrap();
        }

        assert!(streaming.total_buffer_size() > 0);

        streaming.clear_buffers();

        assert_eq!(streaming.total_buffer_size(), 0);
    }

    #[test]
    fn test_data_staleness() {
        let mut data = create_test_data("test", 1);
        data.timestamp = chrono::Utc::now().timestamp_millis() - 10000; // 10 seconds ago

        assert!(data.latency_ms() >= 10000);
    }
}
