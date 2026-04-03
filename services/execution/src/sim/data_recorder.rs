//! Data Recorder for Capturing Live Market Data
//!
//! Records live market data from exchanges to QuestDB for later replay in backtesting.
//! Supports recording ticks, trades, order book snapshots, and candles.
//!
//! ## Architecture
//!
//! ```text
//! Live WebSocket Feeds
//!        │
//!        ▼
//! ┌──────────────┐
//! │ DataRecorder │
//! │  (buffered)  │
//! └──────┬───────┘
//!        │ ILP Protocol
//!        ▼
//! ┌──────────────┐
//! │   QuestDB    │
//! │ (time-series)│
//! └──────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use janus_execution::sim::{DataRecorder, RecorderConfig};
//!
//! let config = RecorderConfig::new("localhost", 9009)
//!     .with_batch_size(1000)
//!     .with_flush_interval(Duration::from_secs(1));
//!
//! let recorder = DataRecorder::new(config).await?;
//! recorder.start().await?;
//!
//! // Record events from data feed
//! recorder.record_event(&market_event).await?;
//!
//! // Or subscribe to a data feed
//! recorder.subscribe_to_feed(&data_feed).await?;
//! ```

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use rust_decimal::prelude::ToPrimitive;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;
use thiserror::Error;
use tokio::io::{AsyncWriteExt, BufWriter};
use tokio::net::TcpStream;
use tokio::sync::mpsc;
use tokio::time::interval;
use tracing::{debug, error, info, warn};

use super::data_feed::{
    AggregatedDataFeed, CandleData, DataFeed, MarketEvent, OrderBookData, TickData, TradeData,
    TradeSide,
};
use super::local_fallback::{LocalFallbackConfig, LocalFallbackWriter};

/// Errors that can occur during recording
#[derive(Debug, Error)]
pub enum RecorderError {
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),

    #[error("Write failed: {0}")]
    WriteFailed(String),

    #[error("Buffer overflow: {0} events dropped")]
    BufferOverflow(usize),

    #[error("Channel closed")]
    ChannelClosed,

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Recorder not running")]
    NotRunning,
}

/// Configuration for the data recorder
#[derive(Debug, Clone)]
pub struct RecorderConfig {
    /// QuestDB host
    pub host: String,
    /// QuestDB ILP port (default: 9009)
    pub port: u16,
    /// Batch size before flushing
    pub batch_size: usize,
    /// Maximum flush interval
    pub flush_interval: Duration,
    /// Channel buffer size
    pub channel_buffer: usize,
    /// Reconnect delay on failure
    pub reconnect_delay: Duration,
    /// Maximum reconnect attempts (0 = infinite)
    pub max_reconnect_attempts: u32,
    /// Record tick data
    pub record_ticks: bool,
    /// Record trade data
    pub record_trades: bool,
    /// Record order book data
    pub record_orderbook: bool,
    /// Record candle data
    pub record_candles: bool,
    /// Table name prefix
    pub table_prefix: String,
    /// Local fallback configuration (optional)
    /// When enabled, events are written to local disk when QuestDB is unavailable
    pub fallback_config: Option<LocalFallbackConfig>,
    /// Write to fallback concurrently with QuestDB (dual-write mode)
    /// When false, fallback is only used when QuestDB connection fails
    pub dual_write_fallback: bool,
}

impl Default for RecorderConfig {
    fn default() -> Self {
        Self {
            host: "localhost".to_string(),
            port: 9009,
            batch_size: 1000,
            flush_interval: Duration::from_secs(1),
            channel_buffer: 100_000,
            reconnect_delay: Duration::from_secs(5),
            max_reconnect_attempts: 0, // Infinite
            record_ticks: true,
            record_trades: true,
            record_orderbook: false, // Off by default (high volume)
            record_candles: true,
            table_prefix: "fks".to_string(),
            fallback_config: None,
            dual_write_fallback: false,
        }
    }
}

impl RecorderConfig {
    /// Create a new recorder configuration
    pub fn new(host: &str, port: u16) -> Self {
        Self {
            host: host.to_string(),
            port,
            ..Default::default()
        }
    }

    /// Set batch size
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Set flush interval
    pub fn with_flush_interval(mut self, interval: Duration) -> Self {
        self.flush_interval = interval;
        self
    }

    /// Set channel buffer size
    pub fn with_channel_buffer(mut self, size: usize) -> Self {
        self.channel_buffer = size;
        self
    }

    /// Set reconnect delay
    pub fn with_reconnect_delay(mut self, delay: Duration) -> Self {
        self.reconnect_delay = delay;
        self
    }

    /// Set table prefix
    pub fn with_table_prefix(mut self, prefix: &str) -> Self {
        self.table_prefix = prefix.to_string();
        self
    }

    /// Enable/disable tick recording
    pub fn with_record_ticks(mut self, enabled: bool) -> Self {
        self.record_ticks = enabled;
        self
    }

    /// Enable/disable trade recording
    pub fn with_record_trades(mut self, enabled: bool) -> Self {
        self.record_trades = enabled;
        self
    }

    /// Enable/disable order book recording
    pub fn with_record_orderbook(mut self, enabled: bool) -> Self {
        self.record_orderbook = enabled;
        self
    }

    /// Enable/disable candle recording
    pub fn with_record_candles(mut self, enabled: bool) -> Self {
        self.record_candles = enabled;
        self
    }

    /// Get the QuestDB connection string
    pub fn connection_string(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }

    /// Get table name for ticks
    pub fn ticks_table(&self) -> String {
        format!("{}_ticks", self.table_prefix)
    }

    /// Get table name for trades
    pub fn trades_table(&self) -> String {
        format!("{}_trades", self.table_prefix)
    }

    /// Get table name for order book
    pub fn orderbook_table(&self) -> String {
        format!("{}_orderbook", self.table_prefix)
    }

    /// Get table name for candles
    pub fn candles_table(&self) -> String {
        format!("{}_candles", self.table_prefix)
    }

    /// Set local fallback configuration
    pub fn with_fallback(mut self, config: LocalFallbackConfig) -> Self {
        self.fallback_config = Some(config);
        self
    }

    /// Enable dual-write mode (write to both QuestDB and fallback concurrently)
    pub fn with_dual_write(mut self, enabled: bool) -> Self {
        self.dual_write_fallback = enabled;
        self
    }

    /// Check if fallback is configured
    pub fn has_fallback(&self) -> bool {
        self.fallback_config.as_ref().is_some_and(|c| c.enabled)
    }
}

/// Recording statistics
#[derive(Debug, Clone, Default)]
pub struct RecorderStats {
    /// Total events recorded
    pub events_recorded: u64,
    /// Ticks recorded
    pub ticks_recorded: u64,
    /// Trades recorded
    pub trades_recorded: u64,
    /// Order book snapshots recorded
    pub orderbooks_recorded: u64,
    /// Candles recorded
    pub candles_recorded: u64,
    /// Events dropped (buffer overflow)
    pub events_dropped: u64,
    /// Write errors
    pub write_errors: u64,
    /// Reconnection count
    pub reconnections: u64,
    /// Start time
    pub start_time: Option<DateTime<Utc>>,
    /// Last write time
    pub last_write_time: Option<DateTime<Utc>>,
    /// Current buffer depth (pending writes)
    pub buffer_depth: usize,
    /// Buffer capacity
    pub buffer_capacity: usize,
    /// Channel queue depth (pending events in channel)
    pub channel_depth: usize,
    /// Channel capacity
    pub channel_capacity: usize,
    /// Whether currently connected to QuestDB
    pub connected: bool,
    /// Total bytes written
    pub bytes_written: u64,
    /// Successful flushes
    pub flush_count: u64,
    /// Whether fallback is currently active (writing to disk)
    pub fallback_active: bool,
    /// Events written to fallback
    pub fallback_events: u64,
    /// Bytes written to fallback
    pub fallback_bytes: u64,
    /// Fallback write errors
    pub fallback_errors: u64,
}

impl RecorderStats {
    /// Get events per second
    pub fn events_per_second(&self) -> f64 {
        match self.start_time {
            Some(start) => {
                let duration = (Utc::now() - start).num_milliseconds() as f64 / 1000.0;
                if duration > 0.0 {
                    self.events_recorded as f64 / duration
                } else {
                    0.0
                }
            }
            None => 0.0,
        }
    }

    /// Get buffer utilization percentage
    pub fn buffer_utilization_pct(&self) -> f64 {
        if self.buffer_capacity > 0 {
            (self.buffer_depth as f64 / self.buffer_capacity as f64) * 100.0
        } else {
            0.0
        }
    }

    /// Get channel utilization percentage
    pub fn channel_utilization_pct(&self) -> f64 {
        if self.channel_capacity > 0 {
            (self.channel_depth as f64 / self.channel_capacity as f64) * 100.0
        } else {
            0.0
        }
    }

    /// Get drop rate percentage
    pub fn drop_rate_pct(&self) -> f64 {
        let total = self.events_recorded + self.events_dropped;
        if total > 0 {
            (self.events_dropped as f64 / total as f64) * 100.0
        } else {
            0.0
        }
    }

    /// Get error rate percentage
    pub fn error_rate_pct(&self) -> f64 {
        if self.flush_count > 0 {
            (self.write_errors as f64 / self.flush_count as f64) * 100.0
        } else {
            0.0
        }
    }

    /// Check if recorder is healthy (connected, low drop rate, low error rate)
    pub fn is_healthy(&self) -> bool {
        self.connected && self.drop_rate_pct() < 1.0 && self.error_rate_pct() < 5.0
    }

    /// Check if fallback is being used (QuestDB unavailable)
    pub fn using_fallback(&self) -> bool {
        self.fallback_active && !self.connected
    }

    /// Get fallback utilization info
    pub fn fallback_info(&self) -> Option<(u64, u64)> {
        if self.fallback_events > 0 || self.fallback_bytes > 0 {
            Some((self.fallback_events, self.fallback_bytes))
        } else {
            None
        }
    }

    /// Get uptime in seconds
    pub fn uptime_seconds(&self) -> f64 {
        match self.start_time {
            Some(start) => (Utc::now() - start).num_milliseconds() as f64 / 1000.0,
            None => 0.0,
        }
    }
}

/// Message types for the recording channel
enum RecordMessage {
    Event(MarketEvent),
    Flush,
    Shutdown,
}

/// Data recorder for capturing market data to QuestDB
pub struct DataRecorder {
    /// Configuration
    config: RecorderConfig,
    /// Channel sender for events
    tx: Option<mpsc::Sender<RecordMessage>>,
    /// Statistics
    stats: Arc<RwLock<RecorderStats>>,
    /// Running flag
    running: Arc<AtomicBool>,
    /// Events recorded counter (atomic for fast access)
    events_counter: Arc<AtomicU64>,
    /// Task handle for writer loop
    task_handle: Option<tokio::task::JoinHandle<()>>,
    /// Task handles for feed subscriptions
    subscription_handles: Vec<tokio::task::JoinHandle<()>>,
}

impl DataRecorder {
    /// Create a new data recorder
    pub fn new(config: RecorderConfig) -> Self {
        let channel_capacity = config.channel_buffer;
        Self {
            config,
            tx: None,
            stats: Arc::new(RwLock::new(RecorderStats {
                channel_capacity,
                buffer_capacity: 0, // Set when writer starts
                ..Default::default()
            })),
            running: Arc::new(AtomicBool::new(false)),
            events_counter: Arc::new(AtomicU64::new(0)),
            task_handle: None,
            subscription_handles: Vec::new(),
        }
    }

    /// Start the recorder
    pub async fn start(&mut self) -> Result<(), RecorderError> {
        if self.running.load(Ordering::SeqCst) {
            return Ok(());
        }

        info!(
            "Starting data recorder, connecting to {}",
            self.config.connection_string()
        );

        // Create channel
        let (tx, rx) = mpsc::channel(self.config.channel_buffer);
        self.tx = Some(tx);

        // Initialize stats
        {
            let mut stats = self.stats.write();
            stats.start_time = Some(Utc::now());
        }

        // Spawn writer task
        let config = self.config.clone();
        let stats = self.stats.clone();
        let running = self.running.clone();
        let events_counter = self.events_counter.clone();

        running.store(true, Ordering::SeqCst);

        let handle = tokio::spawn(async move {
            writer_loop(config, rx, stats, running, events_counter).await;
        });

        self.task_handle = Some(handle);

        info!("Data recorder started");
        Ok(())
    }

    /// Stop the recorder
    pub async fn stop(&mut self) -> Result<(), RecorderError> {
        if !self.running.load(Ordering::SeqCst) {
            return Ok(());
        }

        info!("Stopping data recorder");

        // Send shutdown signal
        if let Some(tx) = &self.tx {
            let _ = tx.send(RecordMessage::Shutdown).await;
        }

        // Wait for task to complete
        if let Some(handle) = self.task_handle.take() {
            let _ = handle.await;
        }

        self.running.store(false, Ordering::SeqCst);
        self.tx = None;

        info!("Data recorder stopped");
        Ok(())
    }

    /// Record a market event
    pub async fn record_event(&self, event: &MarketEvent) -> Result<(), RecorderError> {
        // Check if we should record this event type
        let should_record = match event {
            MarketEvent::Tick(_) => self.config.record_ticks,
            MarketEvent::Trade(_) => self.config.record_trades,
            MarketEvent::OrderBook(_) => self.config.record_orderbook,
            MarketEvent::Candle(_) => self.config.record_candles,
            _ => false,
        };

        if !should_record {
            return Ok(());
        }

        let tx = self.tx.as_ref().ok_or(RecorderError::NotRunning)?;

        tx.send(RecordMessage::Event(event.clone()))
            .await
            .map_err(|_| RecorderError::ChannelClosed)?;

        Ok(())
    }

    /// Force flush buffered data
    pub async fn flush(&self) -> Result<(), RecorderError> {
        let tx = self.tx.as_ref().ok_or(RecorderError::NotRunning)?;

        tx.send(RecordMessage::Flush)
            .await
            .map_err(|_| RecorderError::ChannelClosed)?;

        Ok(())
    }

    /// Get current statistics
    pub fn stats(&self) -> RecorderStats {
        let mut stats = self.stats.read().clone();
        // Update events_recorded from atomic counter
        stats.events_recorded = self.events_counter.load(Ordering::Relaxed);
        stats
    }

    /// Check if recorder is running
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Get configuration
    pub fn config(&self) -> &RecorderConfig {
        &self.config
    }

    /// Subscribe to an AggregatedDataFeed and automatically record all events
    ///
    /// This spawns a background task that listens to the feed and records
    /// all events matching the recorder's configuration.
    ///
    /// # Arguments
    ///
    /// * `feed` - The data feed to subscribe to
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let feed = AggregatedDataFeed::new("live");
    /// let mut recorder = DataRecorder::new(config);
    /// recorder.start().await?;
    /// recorder.subscribe_to_feed(&feed).await?;
    /// ```
    pub async fn subscribe_to_feed(
        &mut self,
        feed: &AggregatedDataFeed,
    ) -> Result<(), RecorderError> {
        if !self.running.load(Ordering::SeqCst) {
            return Err(RecorderError::NotRunning);
        }

        let tx = self.tx.as_ref().ok_or(RecorderError::NotRunning)?.clone();
        let mut rx = feed.subscribe();
        let running = self.running.clone();
        let config = self.config.clone();
        let stats = self.stats.clone();

        info!("Subscribing data recorder to feed: {}", feed.name());

        let handle = tokio::spawn(async move {
            loop {
                if !running.load(Ordering::SeqCst) {
                    break;
                }

                match rx.recv().await {
                    Ok(event) => {
                        // Check if we should record this event type
                        let should_record = match &event {
                            MarketEvent::Tick(_) => config.record_ticks,
                            MarketEvent::Trade(_) => config.record_trades,
                            MarketEvent::OrderBook(_) => config.record_orderbook,
                            MarketEvent::Candle(_) => config.record_candles,
                            _ => false,
                        };

                        if should_record {
                            if let Err(e) = tx.send(RecordMessage::Event(event)).await {
                                warn!("Failed to send event to recorder: {}", e);
                                stats.write().events_dropped += 1;
                            }
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                        warn!("Recorder feed subscription lagged by {} events", n);
                        stats.write().events_dropped += n;
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                        info!("Feed channel closed, stopping recorder subscription");
                        break;
                    }
                }
            }
            debug!("Feed subscription task exited");
        });

        self.subscription_handles.push(handle);
        Ok(())
    }

    /// Subscribe to a SimEnvironment's data feed
    ///
    /// Convenience method for recording from a SimEnvironment.
    pub async fn subscribe_to_sim_feed(
        &mut self,
        data_feed: &Arc<parking_lot::RwLock<AggregatedDataFeed>>,
    ) -> Result<(), RecorderError> {
        let feed = data_feed.read();
        self.subscribe_to_feed(&feed).await
    }

    /// Get the number of active feed subscriptions
    pub fn subscription_count(&self) -> usize {
        self.subscription_handles
            .iter()
            .filter(|h| !h.is_finished())
            .count()
    }

    /// Stop all feed subscriptions (but keep the recorder running)
    pub fn stop_subscriptions(&mut self) {
        for handle in self.subscription_handles.drain(..) {
            handle.abort();
        }
    }
}

impl Drop for DataRecorder {
    fn drop(&mut self) {
        if self.running.load(Ordering::SeqCst) {
            warn!("DataRecorder dropped while still running");
            self.running.store(false, Ordering::SeqCst);

            // Abort subscription tasks
            for handle in self.subscription_handles.drain(..) {
                handle.abort();
            }

            // Abort writer task
            if let Some(handle) = self.task_handle.take() {
                handle.abort();
            }
        }
    }
}

/// Writer loop that handles buffering and flushing to QuestDB with fallback support
async fn writer_loop(
    config: RecorderConfig,
    mut rx: mpsc::Receiver<RecordMessage>,
    stats: Arc<RwLock<RecorderStats>>,
    running: Arc<AtomicBool>,
    events_counter: Arc<AtomicU64>,
) {
    let buffer_capacity = config.batch_size * 2;
    let mut buffer: VecDeque<String> = VecDeque::with_capacity(buffer_capacity);
    // Also keep raw events for fallback writing (if enabled)
    let mut event_buffer: VecDeque<MarketEvent> = VecDeque::with_capacity(buffer_capacity);
    let mut flush_timer = interval(config.flush_interval);
    let mut reconnect_attempts: u32 = 0;

    // Set buffer capacity in stats
    {
        let mut s = stats.write();
        s.buffer_capacity = buffer_capacity;
    }

    // Initialize fallback writer if configured
    let mut fallback_writer: Option<LocalFallbackWriter> = None;
    if let Some(ref fallback_config) = config.fallback_config {
        if fallback_config.enabled {
            match LocalFallbackWriter::new(fallback_config.clone()) {
                Ok(mut writer) => {
                    if let Err(e) = writer.start() {
                        error!("Failed to start fallback writer: {}", e);
                    } else {
                        info!("Fallback writer started at {:?}", fallback_config.base_path);
                        fallback_writer = Some(writer);
                    }
                }
                Err(e) => {
                    error!("Failed to create fallback writer: {}", e);
                }
            }
        }
    }
    let dual_write = config.dual_write_fallback;

    // Initial connection
    let mut connection = connect_to_questdb(&config).await;
    {
        let mut s = stats.write();
        s.connected = connection.is_some();
    }
    if connection.is_some() {
        info!("Connected to QuestDB at {}", config.connection_string());
    } else if fallback_writer.is_some() {
        warn!("QuestDB unavailable, using fallback storage");
        stats.write().fallback_active = true;
    }

    loop {
        // Update buffer depth and fallback stats periodically
        {
            let mut s = stats.write();
            s.buffer_depth = buffer.len();
            s.connected = connection.is_some();
            // Update fallback active status
            if let Some(ref fw) = fallback_writer {
                let fb_stats = fw.stats();
                s.fallback_active = fb_stats.active;
                s.fallback_events = fb_stats.events_written;
                s.fallback_bytes = fb_stats.bytes_written;
                s.fallback_errors = fb_stats.write_errors;
            }
        }

        tokio::select! {
            msg = rx.recv() => {
                match msg {
                    Some(RecordMessage::Event(event)) => {
                        // Format and buffer the event
                        if let Some(line) = format_event(&event, &config) {
                            buffer.push_back(line);
                            event_buffer.push_back(event.clone());
                            events_counter.fetch_add(1, Ordering::Relaxed);

                            // Update type-specific counters and buffer depth
                            {
                                let mut s = stats.write();
                                s.buffer_depth = buffer.len();
                                match event {
                                    MarketEvent::Tick(_) => s.ticks_recorded += 1,
                                    MarketEvent::Trade(_) => s.trades_recorded += 1,
                                    MarketEvent::OrderBook(_) => s.orderbooks_recorded += 1,
                                    MarketEvent::Candle(_) => s.candles_recorded += 1,
                                    _ => {}
                                }
                            }

                            // Flush if batch is full
                            if buffer.len() >= config.batch_size {
                                let flush_result = flush_buffer_with_fallback(
                                    &mut buffer,
                                    &mut event_buffer,
                                    &mut connection,
                                    &config,
                                    &stats,
                                    &mut fallback_writer,
                                    dual_write,
                                ).await;

                                if let Err(e) = flush_result {
                                    warn!("Flush error: {}", e);
                                    // Attempt reconnection
                                    if should_reconnect(&config, reconnect_attempts) {
                                        connection = reconnect(&config, &stats, &mut reconnect_attempts).await;
                                    }
                                }
                            }
                        }
                    }
                    Some(RecordMessage::Flush) => {
                        if let Err(e) = flush_buffer_with_fallback(
                            &mut buffer,
                            &mut event_buffer,
                            &mut connection,
                            &config,
                            &stats,
                            &mut fallback_writer,
                            dual_write,
                        ).await {
                            warn!("Flush error: {}", e);
                        }
                    }
                    Some(RecordMessage::Shutdown) | None => {
                        // Final flush before shutdown
                        if !buffer.is_empty() {
                            let _ = flush_buffer_with_fallback(
                                &mut buffer,
                                &mut event_buffer,
                                &mut connection,
                                &config,
                                &stats,
                                &mut fallback_writer,
                                dual_write,
                            ).await;
                        }
                        // Stop fallback writer
                        if let Some(ref mut fw) = fallback_writer {
                            if let Err(e) = fw.stop().await {
                                warn!("Error stopping fallback writer: {}", e);
                            }
                        }
                        break;
                    }
                }
            }
            _ = flush_timer.tick() => {
                // Periodic flush
                if !buffer.is_empty() {
                    let flush_result = flush_buffer_with_fallback(
                        &mut buffer,
                        &mut event_buffer,
                        &mut connection,
                        &config,
                        &stats,
                        &mut fallback_writer,
                        dual_write,
                    ).await;

                    if let Err(e) = flush_result {
                        warn!("Periodic flush error: {}", e);
                        // Attempt reconnection
                        if should_reconnect(&config, reconnect_attempts) {
                            connection = reconnect(&config, &stats, &mut reconnect_attempts).await;
                        }
                    }
                }
            }
        }
    }

    // Final stats update
    {
        let mut s = stats.write();
        s.buffer_depth = 0;
        s.connected = false;
        s.fallback_active = false;
    }

    running.store(false, Ordering::SeqCst);
    debug!("Writer loop exited");
}

/// Connect to QuestDB
async fn connect_to_questdb(config: &RecorderConfig) -> Option<BufWriter<TcpStream>> {
    match TcpStream::connect(config.connection_string()).await {
        Ok(stream) => {
            // Set TCP_NODELAY for lower latency
            let _ = stream.set_nodelay(true);
            Some(BufWriter::with_capacity(64 * 1024, stream))
        }
        Err(e) => {
            error!("Failed to connect to QuestDB: {}", e);
            None
        }
    }
}

/// Check if we should attempt reconnection
fn should_reconnect(config: &RecorderConfig, attempts: u32) -> bool {
    config.max_reconnect_attempts == 0 || attempts < config.max_reconnect_attempts
}

/// Attempt to reconnect to QuestDB
async fn reconnect(
    config: &RecorderConfig,
    stats: &Arc<RwLock<RecorderStats>>,
    attempts: &mut u32,
) -> Option<BufWriter<TcpStream>> {
    *attempts += 1;
    stats.write().reconnections += 1;

    info!("Attempting reconnection to QuestDB (attempt {})", attempts);

    tokio::time::sleep(config.reconnect_delay).await;

    let conn = connect_to_questdb(config).await;
    if conn.is_some() {
        info!("Reconnected to QuestDB");
        *attempts = 0; // Reset on successful connection
    }
    conn
}

/// Flush buffer to QuestDB with fallback support
async fn flush_buffer_with_fallback(
    buffer: &mut VecDeque<String>,
    event_buffer: &mut VecDeque<MarketEvent>,
    connection: &mut Option<BufWriter<TcpStream>>,
    config: &RecorderConfig,
    stats: &Arc<RwLock<RecorderStats>>,
    fallback_writer: &mut Option<LocalFallbackWriter>,
    dual_write: bool,
) -> Result<(), RecorderError> {
    if buffer.is_empty() {
        return Ok(());
    }

    // In dual-write mode, always write to fallback first
    if dual_write {
        if let Some(fw) = fallback_writer {
            write_events_to_fallback(fw, event_buffer, stats).await;
        }
    }

    // Try to write to QuestDB
    let questdb_result = flush_buffer_to_questdb(buffer, connection, config, stats).await;

    // If QuestDB write failed and we're not in dual-write mode, try fallback
    if questdb_result.is_err() && !dual_write {
        if let Some(fw) = fallback_writer {
            info!("QuestDB unavailable, writing to fallback storage");
            stats.write().fallback_active = true;
            write_events_to_fallback(fw, event_buffer, stats).await;
            // Clear the ILP buffer since we wrote to fallback
            buffer.clear();
            // Return Ok since we successfully wrote to fallback
            return Ok(());
        }
    }

    // Clear event buffer after processing
    event_buffer.clear();

    questdb_result
}

/// Write events to fallback storage
async fn write_events_to_fallback(
    fallback_writer: &mut LocalFallbackWriter,
    event_buffer: &mut VecDeque<MarketEvent>,
    stats: &Arc<RwLock<RecorderStats>>,
) {
    for event in event_buffer.drain(..) {
        if let Err(e) = fallback_writer.write_event(&event).await {
            warn!("Fallback write error: {}", e);
            stats.write().fallback_errors += 1;
        }
    }
    // Request flush
    if let Err(e) = fallback_writer.flush().await {
        warn!("Fallback flush error: {}", e);
    }
}

/// Flush buffer to QuestDB (original logic)
async fn flush_buffer_to_questdb(
    buffer: &mut VecDeque<String>,
    connection: &mut Option<BufWriter<TcpStream>>,
    config: &RecorderConfig,
    stats: &Arc<RwLock<RecorderStats>>,
) -> Result<(), RecorderError> {
    if buffer.is_empty() {
        return Ok(());
    }

    let conn = match connection {
        Some(c) => c,
        None => {
            // Try to reconnect
            *connection = connect_to_questdb(config).await;
            {
                let mut s = stats.write();
                s.connected = connection.is_some();
            }
            match connection {
                Some(c) => c,
                None => {
                    // Can't connect - don't drop events here, let caller handle fallback
                    return Err(RecorderError::ConnectionFailed(
                        "Not connected to QuestDB".to_string(),
                    ));
                }
            }
        }
    };

    // Write all buffered lines
    let mut write_error = false;
    let mut bytes_written: u64 = 0;
    let mut lines_written = 0;
    while let Some(line) = buffer.pop_front() {
        let line_bytes = line.as_bytes();
        if let Err(e) = conn.write_all(line_bytes).await {
            error!("Write error: {}", e);
            stats.write().write_errors += 1;
            write_error = true;
            break;
        }
        bytes_written += line_bytes.len() as u64;
        lines_written += 1;
    }

    // Flush the writer
    if !write_error {
        if let Err(e) = conn.flush().await {
            error!("Flush error: {}", e);
            stats.write().write_errors += 1;
            write_error = true;
        }
    }

    if write_error {
        // Connection is bad, close it
        *connection = None;
        let mut s = stats.write();
        s.connected = false;
        return Err(RecorderError::WriteFailed("Write failed".to_string()));
    }

    // Update stats on successful flush
    {
        let mut s = stats.write();
        s.last_write_time = Some(Utc::now());
        s.bytes_written += bytes_written;
        s.flush_count += 1;
        s.buffer_depth = buffer.len();
        // Clear fallback_active if we successfully wrote to QuestDB
        s.fallback_active = false;
    }

    debug!(
        "Flushed {} lines ({} bytes) to QuestDB",
        lines_written, bytes_written
    );
    Ok(())
}

/// Format a market event as QuestDB ILP line protocol
fn format_event(event: &MarketEvent, config: &RecorderConfig) -> Option<String> {
    match event {
        MarketEvent::Tick(tick) => Some(format_tick(tick, &config.ticks_table())),
        MarketEvent::Trade(trade) => Some(format_trade(trade, &config.trades_table())),
        MarketEvent::OrderBook(ob) => Some(format_orderbook(ob, &config.orderbook_table())),
        MarketEvent::Candle(candle) => Some(format_candle(candle, &config.candles_table())),
        _ => None,
    }
}

/// Format tick data as ILP line
fn format_tick(tick: &TickData, table: &str) -> String {
    // ILP format: table,symbol=X,exchange=Y bid=1.0,ask=2.0,bid_size=3.0,ask_size=4.0 timestamp
    format!(
        "{},symbol={},exchange={} bid={},ask={},bid_size={},ask_size={},spread_bps={} {}\n",
        table,
        escape_tag(&tick.symbol),
        escape_tag(&tick.exchange),
        tick.bid_price.to_f64().unwrap_or(0.0),
        tick.ask_price.to_f64().unwrap_or(0.0),
        tick.bid_size.to_f64().unwrap_or(0.0),
        tick.ask_size.to_f64().unwrap_or(0.0),
        tick.spread_bps().to_f64().unwrap_or(0.0),
        tick.timestamp.timestamp_nanos_opt().unwrap_or(0)
    )
}

/// Format trade data as ILP line
fn format_trade(trade: &TradeData, table: &str) -> String {
    let side_str = match trade.side {
        TradeSide::Buy => "buy",
        TradeSide::Sell => "sell",
        TradeSide::Unknown => "unknown",
    };

    format!(
        "{},symbol={},exchange={},side={} price={},size={},trade_id=\"{}\" {}\n",
        table,
        escape_tag(&trade.symbol),
        escape_tag(&trade.exchange),
        side_str,
        trade.price.to_f64().unwrap_or(0.0),
        trade.size.to_f64().unwrap_or(0.0),
        escape_string(&trade.trade_id),
        trade.timestamp.timestamp_nanos_opt().unwrap_or(0)
    )
}

/// Format order book data as ILP line
fn format_orderbook(ob: &OrderBookData, table: &str) -> String {
    // For order book, we store top N levels
    let max_levels = 5;

    let bids: Vec<String> = ob
        .bids
        .iter()
        .take(max_levels)
        .enumerate()
        .flat_map(|(i, level)| {
            vec![
                format!("bid{}_px={}", i, level.price.to_f64().unwrap_or(0.0)),
                format!("bid{}_sz={}", i, level.size.to_f64().unwrap_or(0.0)),
            ]
        })
        .collect();

    let asks: Vec<String> = ob
        .asks
        .iter()
        .take(max_levels)
        .enumerate()
        .flat_map(|(i, level)| {
            vec![
                format!("ask{}_px={}", i, level.price.to_f64().unwrap_or(0.0)),
                format!("ask{}_sz={}", i, level.size.to_f64().unwrap_or(0.0)),
            ]
        })
        .collect();

    let fields = [bids, asks].concat().join(",");

    format!(
        "{},symbol={},exchange={} {} {}\n",
        table,
        escape_tag(&ob.symbol),
        escape_tag(&ob.exchange),
        fields,
        ob.timestamp.timestamp_nanos_opt().unwrap_or(0)
    )
}

/// Format candle data as ILP line
fn format_candle(candle: &CandleData, table: &str) -> String {
    format!(
        "{},symbol={},exchange={} open={},high={},low={},close={},volume={},trades={} {}\n",
        table,
        escape_tag(&candle.symbol),
        escape_tag(&candle.exchange),
        candle.open.to_f64().unwrap_or(0.0),
        candle.high.to_f64().unwrap_or(0.0),
        candle.low.to_f64().unwrap_or(0.0),
        candle.close.to_f64().unwrap_or(0.0),
        candle.volume.to_f64().unwrap_or(0.0),
        candle.trade_count,
        candle.close_time.timestamp_nanos_opt().unwrap_or(0)
    )
}

/// Escape a tag value for ILP protocol
fn escape_tag(s: &str) -> String {
    s.replace(',', "\\,")
        .replace('=', "\\=")
        .replace(' ', "\\ ")
        .replace('/', "_") // Replace / with _ for symbols like BTC/USDT
}

/// Escape a string value for ILP protocol
fn escape_string(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

/// Get current timestamp in nanoseconds
pub fn now_nanos() -> i64 {
    Utc::now().timestamp_nanos_opt().unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_recorder_config() {
        let config = RecorderConfig::new("localhost", 9009)
            .with_batch_size(500)
            .with_flush_interval(Duration::from_millis(500))
            .with_table_prefix("test");

        assert_eq!(config.host, "localhost");
        assert_eq!(config.port, 9009);
        assert_eq!(config.batch_size, 500);
        assert_eq!(config.table_prefix, "test");
        assert_eq!(config.ticks_table(), "test_ticks");
        assert_eq!(config.trades_table(), "test_trades");
    }

    #[test]
    fn test_format_tick() {
        let tick = TickData::new(
            "BTC/USDT",
            "kraken",
            dec!(50000.0),
            dec!(50010.0),
            dec!(1.5),
            dec!(2.0),
            Utc::now(),
        );

        let line = format_tick(&tick, "fks_ticks");

        assert!(line.starts_with("fks_ticks,symbol=BTC_USDT,exchange=kraken"));
        assert!(line.contains("bid=50000"));
        assert!(line.contains("ask=50010"));
        assert!(line.contains("bid_size=1.5"));
        assert!(line.contains("ask_size=2"));
        assert!(line.ends_with('\n'));
    }

    #[test]
    fn test_format_trade() {
        let trade = TradeData {
            symbol: "ETH/USDT".to_string(),
            exchange: "binance".to_string(),
            price: dec!(3000.0),
            size: dec!(10.0),
            side: TradeSide::Buy,
            trade_id: "12345".to_string(),
            timestamp: Utc::now(),
        };

        let line = format_trade(&trade, "fks_trades");

        assert!(line.starts_with("fks_trades,symbol=ETH_USDT,exchange=binance,side=buy"));
        assert!(line.contains("price=3000"));
        assert!(line.contains("size=10"));
        assert!(line.contains("trade_id=\"12345\""));
    }

    #[test]
    fn test_escape_tag() {
        assert_eq!(escape_tag("BTC/USDT"), "BTC_USDT");
        assert_eq!(escape_tag("test,value"), "test\\,value");
        assert_eq!(escape_tag("key=value"), "key\\=value");
        assert_eq!(escape_tag("hello world"), "hello\\ world");
    }

    #[test]
    fn test_escape_string() {
        assert_eq!(escape_string("hello\"world"), "hello\\\"world");
        assert_eq!(escape_string("line1\nline2"), "line1\\nline2");
        assert_eq!(escape_string("path\\to\\file"), "path\\\\to\\\\file");
    }

    #[test]
    fn test_recorder_stats() {
        let mut stats = RecorderStats::default();
        stats.start_time = Some(Utc::now());
        stats.events_recorded = 1000;

        // events_per_second should be > 0 since we just started
        assert!(stats.events_per_second() >= 0.0);
    }

    #[tokio::test]
    async fn test_recorder_creation() {
        let config = RecorderConfig::new("localhost", 9009);
        let recorder = DataRecorder::new(config);

        assert!(!recorder.is_running());
        assert_eq!(recorder.stats().events_recorded, 0);
    }

    #[test]
    fn test_format_candle() {
        let candle = CandleData {
            symbol: "BTC/USDT".to_string(),
            exchange: "bybit".to_string(),
            open_time: Utc::now(),
            close_time: Utc::now(),
            open: dec!(50000.0),
            high: dec!(50100.0),
            low: dec!(49900.0),
            close: dec!(50050.0),
            volume: dec!(100.0),
            trade_count: 500,
        };

        let line = format_candle(&candle, "fks_candles");

        assert!(line.starts_with("fks_candles,symbol=BTC_USDT,exchange=bybit"));
        assert!(line.contains("open=50000"));
        assert!(line.contains("high=50100"));
        assert!(line.contains("low=49900"));
        assert!(line.contains("close=50050"));
        assert!(line.contains("volume=100"));
        assert!(line.contains("trades=500"));
    }

    #[test]
    fn test_recorder_stats_buffer_utilization() {
        let mut stats = RecorderStats::default();
        stats.buffer_capacity = 1000;
        stats.buffer_depth = 250;

        assert!((stats.buffer_utilization_pct() - 25.0).abs() < 0.01);
    }

    #[test]
    fn test_recorder_stats_channel_utilization() {
        let mut stats = RecorderStats::default();
        stats.channel_capacity = 10000;
        stats.channel_depth = 5000;

        assert!((stats.channel_utilization_pct() - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_recorder_stats_drop_rate() {
        let mut stats = RecorderStats::default();
        stats.events_recorded = 990;
        stats.events_dropped = 10;

        assert!((stats.drop_rate_pct() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_recorder_stats_error_rate() {
        let mut stats = RecorderStats::default();
        stats.flush_count = 100;
        stats.write_errors = 5;

        assert!((stats.error_rate_pct() - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_recorder_stats_is_healthy() {
        let mut stats = RecorderStats::default();
        stats.connected = true;
        stats.events_recorded = 1000;
        stats.events_dropped = 5; // 0.5% drop rate
        stats.flush_count = 100;
        stats.write_errors = 2; // 2% error rate

        assert!(stats.is_healthy());

        // Not connected - unhealthy
        stats.connected = false;
        assert!(!stats.is_healthy());

        // High drop rate - unhealthy
        stats.connected = true;
        stats.events_dropped = 50; // 5% drop rate
        assert!(!stats.is_healthy());
    }

    #[test]
    fn test_recorder_stats_uptime() {
        let mut stats = RecorderStats::default();
        stats.start_time = Some(Utc::now() - chrono::Duration::seconds(60));

        let uptime = stats.uptime_seconds();
        assert!((59.0..=61.0).contains(&uptime));
    }

    #[tokio::test]
    async fn test_recorder_subscribe_not_running() {
        let config = RecorderConfig::new("localhost", 9009);
        let mut recorder = DataRecorder::new(config);
        let feed = AggregatedDataFeed::new("test");

        // Should fail because recorder is not started
        let result = recorder.subscribe_to_feed(&feed).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_recorder_subscription_count() {
        let config = RecorderConfig::new("localhost", 9009);
        let recorder = DataRecorder::new(config);

        // No subscriptions initially
        assert_eq!(recorder.subscription_count(), 0);
    }

    #[test]
    fn test_recorder_config_with_fallback() {
        use super::super::local_fallback::FallbackFormat;

        let fallback_config = LocalFallbackConfig::new("./data/fallback")
            .with_format(FallbackFormat::NdJson)
            .with_rotation_size_mb(100)
            .with_enabled(true);

        let config = RecorderConfig::new("localhost", 9009)
            .with_fallback(fallback_config)
            .with_dual_write(false);

        assert!(config.has_fallback());
        assert!(!config.dual_write_fallback);
        assert!(config.fallback_config.is_some());
    }

    #[test]
    fn test_recorder_config_no_fallback() {
        let config = RecorderConfig::new("localhost", 9009);

        assert!(!config.has_fallback());
        assert!(config.fallback_config.is_none());
    }

    #[test]
    fn test_recorder_config_dual_write() {
        use super::super::local_fallback::FallbackFormat;

        let fallback_config = LocalFallbackConfig::new("./data/fallback")
            .with_format(FallbackFormat::NdJson)
            .with_enabled(true);

        let config = RecorderConfig::new("localhost", 9009)
            .with_fallback(fallback_config)
            .with_dual_write(true);

        assert!(config.has_fallback());
        assert!(config.dual_write_fallback);
    }

    #[test]
    fn test_recorder_stats_fallback_info() {
        let mut stats = RecorderStats::default();

        // No fallback usage initially
        assert!(stats.fallback_info().is_none());

        // With fallback usage
        stats.fallback_events = 100;
        stats.fallback_bytes = 5000;
        assert_eq!(stats.fallback_info(), Some((100, 5000)));
    }

    #[test]
    fn test_recorder_stats_using_fallback() {
        let mut stats = RecorderStats::default();

        // Not using fallback when connected
        stats.connected = true;
        stats.fallback_active = false;
        assert!(!stats.using_fallback());

        // Not using fallback when active but connected
        stats.connected = true;
        stats.fallback_active = true;
        assert!(!stats.using_fallback());

        // Using fallback when disconnected and active
        stats.connected = false;
        stats.fallback_active = true;
        assert!(stats.using_fallback());

        // Not using fallback when disconnected but not active
        stats.connected = false;
        stats.fallback_active = false;
        assert!(!stats.using_fallback());
    }

    #[tokio::test]
    async fn test_recorder_creation_with_fallback() {
        use super::super::local_fallback::FallbackFormat;

        let fallback_config = LocalFallbackConfig::new("./test_fallback_data")
            .with_format(FallbackFormat::NdJson)
            .with_enabled(true);

        let config = RecorderConfig::new("localhost", 9009).with_fallback(fallback_config);

        let recorder = DataRecorder::new(config);

        assert!(!recorder.is_running());
        assert!(recorder.config().has_fallback());
        let stats = recorder.stats();
        assert!(!stats.fallback_active);
        assert_eq!(stats.fallback_events, 0);
    }

    #[test]
    fn test_recorder_stats_default_fallback_fields() {
        let stats = RecorderStats::default();

        assert!(!stats.fallback_active);
        assert_eq!(stats.fallback_events, 0);
        assert_eq!(stats.fallback_bytes, 0);
        assert_eq!(stats.fallback_errors, 0);
    }
}
