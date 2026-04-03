//! Replay Engine for Historical Data Playback
//!
//! Provides controlled replay of historical market data for backtesting.
//! Supports multiple replay speeds, time-based controls, and event filtering.
//!
//! ## Features
//!
//! - **Variable Speed Replay**: From real-time to maximum speed
//! - **Time Controls**: Pause, resume, seek to specific timestamps
//! - **Event Filtering**: Filter by symbol, exchange, event type
//! - **Progress Tracking**: Monitor replay progress and statistics
//!
//! ## Architecture
//!
//! ```text
//! Historical Data Source
//!         │
//!         ▼
//! ┌───────────────┐
//! │  DataLoader   │
//! │ (Parquet/CSV) │
//! └───────┬───────┘
//!         │
//!         ▼
//! ┌───────────────┐
//! │ ReplayEngine  │
//! │  (time sync)  │
//! └───────┬───────┘
//!         │ controlled rate
//!         ▼
//! ┌───────────────┐
//! │  DataFeed     │
//! │ (broadcast)   │
//! └───────────────┘
//! ```

use chrono::{DateTime, Duration, TimeZone, Utc};
use parking_lot::RwLock;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::str::FromStr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use thiserror::Error;
use tokio::sync::broadcast;
use tokio::time::{Instant, sleep};
use tracing::{debug, info, warn};
use url::form_urlencoded;

use super::data_feed::{MarketEvent, TickData, TradeData, TradeSide};

/// Errors that can occur during replay
#[derive(Debug, Error)]
pub enum ReplayError {
    #[error("Failed to load data: {0}")]
    LoadFailed(String),

    #[error("Invalid data format: {0}")]
    InvalidFormat(String),

    #[error("Seek failed: {0}")]
    SeekFailed(String),

    #[error("Replay not started")]
    NotStarted,

    #[error("Replay already running")]
    AlreadyRunning,

    #[error("End of data")]
    EndOfData,

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("QuestDB query error: {0}")]
    QueryError(String),

    #[error("Parse error: {0}")]
    ParseError(String),
}

/// Replay speed options
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ReplaySpeed {
    /// Real-time (1x speed)
    RealTime,
    /// Fixed multiplier (e.g., 2x, 10x, 100x)
    Multiplier(f64),
    /// As fast as possible
    Maximum,
    /// Fixed events per second
    EventsPerSecond(u32),
}

impl Default for ReplaySpeed {
    fn default() -> Self {
        Self::Maximum
    }
}

impl std::fmt::Display for ReplaySpeed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReplaySpeed::RealTime => write!(f, "1x (real-time)"),
            ReplaySpeed::Multiplier(m) => write!(f, "{}x", m),
            ReplaySpeed::Maximum => write!(f, "max"),
            ReplaySpeed::EventsPerSecond(eps) => write!(f, "{} events/sec", eps),
        }
    }
}

/// Configuration for the replay engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayConfig {
    /// Replay speed
    pub speed: ReplaySpeed,
    /// Start time filter (replay events >= this time)
    pub start_time: Option<DateTime<Utc>>,
    /// End time filter (replay events <= this time)
    pub end_time: Option<DateTime<Utc>>,
    /// Symbol filter (only replay these symbols)
    pub symbols: Option<Vec<String>>,
    /// Exchange filter (only replay from these exchanges)
    pub exchanges: Option<Vec<String>>,
    /// Event type filter
    pub event_types: EventTypeFilter,
    /// Broadcast channel buffer size
    pub channel_buffer: usize,
    /// Progress update interval (in events)
    pub progress_interval: u64,
    /// Enable verbose logging
    pub verbose: bool,
}

impl Default for ReplayConfig {
    fn default() -> Self {
        Self {
            speed: ReplaySpeed::Maximum,
            start_time: None,
            end_time: None,
            symbols: None,
            exchanges: None,
            event_types: EventTypeFilter::default(),
            channel_buffer: 10_000,
            progress_interval: 10_000,
            verbose: false,
        }
    }
}

impl ReplayConfig {
    /// Create a new replay config with maximum speed
    pub fn fast() -> Self {
        Self::default()
    }

    /// Create a replay config for real-time replay
    pub fn real_time() -> Self {
        Self {
            speed: ReplaySpeed::RealTime,
            ..Default::default()
        }
    }

    /// Set replay speed
    pub fn with_speed(mut self, speed: ReplaySpeed) -> Self {
        self.speed = speed;
        self
    }

    /// Set time range filter
    pub fn with_time_range(
        mut self,
        start: Option<DateTime<Utc>>,
        end: Option<DateTime<Utc>>,
    ) -> Self {
        self.start_time = start;
        self.end_time = end;
        self
    }

    /// Set symbol filter
    pub fn with_symbols(mut self, symbols: Vec<String>) -> Self {
        self.symbols = Some(symbols);
        self
    }

    /// Set exchange filter
    pub fn with_exchanges(mut self, exchanges: Vec<String>) -> Self {
        self.exchanges = Some(exchanges);
        self
    }

    /// Set event type filter
    pub fn with_event_types(mut self, filter: EventTypeFilter) -> Self {
        self.event_types = filter;
        self
    }

    /// Set channel buffer size
    pub fn with_channel_buffer(mut self, size: usize) -> Self {
        self.channel_buffer = size;
        self
    }

    /// Enable verbose logging
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }
}

/// Event type filter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventTypeFilter {
    /// Include tick events
    pub ticks: bool,
    /// Include trade events
    pub trades: bool,
    /// Include order book events
    pub orderbook: bool,
    /// Include candle events
    pub candles: bool,
}

impl Default for EventTypeFilter {
    fn default() -> Self {
        Self {
            ticks: true,
            trades: true,
            orderbook: true,
            candles: true,
        }
    }
}

impl EventTypeFilter {
    /// Only ticks
    pub fn ticks_only() -> Self {
        Self {
            ticks: true,
            trades: false,
            orderbook: false,
            candles: false,
        }
    }

    /// Only trades
    pub fn trades_only() -> Self {
        Self {
            ticks: false,
            trades: true,
            orderbook: false,
            candles: false,
        }
    }

    /// Ticks and trades
    pub fn ticks_and_trades() -> Self {
        Self {
            ticks: true,
            trades: true,
            orderbook: false,
            candles: false,
        }
    }

    /// Check if an event should be included
    pub fn should_include(&self, event: &MarketEvent) -> bool {
        match event {
            MarketEvent::Tick(_) => self.ticks,
            MarketEvent::Trade(_) => self.trades,
            MarketEvent::OrderBook(_) => self.orderbook,
            MarketEvent::Candle(_) => self.candles,
            _ => true,
        }
    }
}

/// Replay statistics
#[derive(Debug, Clone, Default)]
pub struct ReplayStats {
    /// Total events loaded
    pub total_events: u64,
    /// Events replayed
    pub events_replayed: u64,
    /// Events skipped (filtered out)
    pub events_skipped: u64,
    /// Current replay position (event index)
    pub current_index: u64,
    /// Current replay time
    pub current_time: Option<DateTime<Utc>>,
    /// Data start time
    pub data_start_time: Option<DateTime<Utc>>,
    /// Data end time
    pub data_end_time: Option<DateTime<Utc>>,
    /// Replay start wall clock time
    pub replay_started: Option<Instant>,
    /// Replay elapsed wall clock duration
    pub elapsed_seconds: f64,
    /// Effective events per second
    pub events_per_second: f64,
    /// Symbols in data
    pub symbols: Vec<String>,
    /// Exchanges in data
    pub exchanges: Vec<String>,
}

impl ReplayStats {
    /// Calculate progress percentage
    pub fn progress_pct(&self) -> f64 {
        if self.total_events > 0 {
            (self.current_index as f64 / self.total_events as f64) * 100.0
        } else {
            0.0
        }
    }

    /// Calculate simulated time elapsed
    pub fn simulated_elapsed(&self) -> Option<Duration> {
        match (self.data_start_time, self.current_time) {
            (Some(start), Some(current)) => Some(current - start),
            _ => None,
        }
    }

    /// Get estimated time remaining (based on current rate)
    pub fn estimated_remaining_seconds(&self) -> Option<f64> {
        if self.events_per_second > 0.0 && self.total_events > self.current_index {
            let remaining_events = self.total_events - self.current_index;
            Some(remaining_events as f64 / self.events_per_second)
        } else {
            None
        }
    }
}

/// Replay state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayState {
    /// Idle, not started
    Idle,
    /// Currently replaying
    Running,
    /// Paused
    Paused,
    /// Completed
    Completed,
    /// Stopped by user
    Stopped,
}

/// Historical market event with loaded data
#[derive(Debug, Clone)]
pub struct HistoricalEvent {
    /// The market event
    pub event: MarketEvent,
    /// Original index in data source
    pub index: u64,
}

/// Replay engine for historical data playback
pub struct ReplayEngine {
    /// Configuration
    config: ReplayConfig,
    /// Loaded events (sorted by timestamp)
    events: Vec<HistoricalEvent>,
    /// Current replay position
    position: Arc<AtomicU64>,
    /// Event broadcaster
    event_tx: broadcast::Sender<MarketEvent>,
    /// Statistics
    stats: Arc<RwLock<ReplayStats>>,
    /// Current state
    state: Arc<RwLock<ReplayState>>,
    /// Running flag
    running: Arc<AtomicBool>,
    /// Paused flag
    paused: Arc<AtomicBool>,
}

impl ReplayEngine {
    /// Create a new replay engine with configuration
    pub fn new(config: ReplayConfig) -> Self {
        let (event_tx, _) = broadcast::channel(config.channel_buffer);

        Self {
            config,
            events: Vec::new(),
            position: Arc::new(AtomicU64::new(0)),
            event_tx,
            stats: Arc::new(RwLock::new(ReplayStats::default())),
            state: Arc::new(RwLock::new(ReplayState::Idle)),
            running: Arc::new(AtomicBool::new(false)),
            paused: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Load events from a vector (used when data is already loaded)
    pub fn load_events(&mut self, events: Vec<MarketEvent>) -> Result<(), ReplayError> {
        info!("Loading {} events into replay engine", events.len());

        // Convert to historical events with index
        let mut historical: Vec<HistoricalEvent> = events
            .into_iter()
            .enumerate()
            .map(|(i, event)| HistoricalEvent {
                event,
                index: i as u64,
            })
            .collect();

        // Sort by timestamp
        historical.sort_by(|a, b| {
            let ts_a = a.event.timestamp();
            let ts_b = b.event.timestamp();
            ts_a.cmp(&ts_b)
        });

        // Apply filters
        let filtered: Vec<HistoricalEvent> = historical
            .into_iter()
            .filter(|h| self.should_include(&h.event))
            .collect();

        // Update statistics
        {
            let mut stats = self.stats.write();
            stats.total_events = filtered.len() as u64;
            stats.events_skipped = 0;

            if let Some(first) = filtered.first() {
                stats.data_start_time = first.event.timestamp();
            }
            if let Some(last) = filtered.last() {
                stats.data_end_time = last.event.timestamp();
            }

            // Collect unique symbols and exchanges
            let mut symbols: std::collections::HashSet<String> = std::collections::HashSet::new();
            let mut exchanges: std::collections::HashSet<String> = std::collections::HashSet::new();

            for h in &filtered {
                if let Some(s) = h.event.symbol() {
                    symbols.insert(s.to_string());
                }
                if let Some(e) = h.event.exchange() {
                    exchanges.insert(e.to_string());
                }
            }

            stats.symbols = symbols.into_iter().collect();
            stats.exchanges = exchanges.into_iter().collect();
        }

        self.events = filtered;
        self.position.store(0, Ordering::SeqCst);

        info!(
            "Loaded {} events, time range: {:?} to {:?}",
            self.events.len(),
            self.stats.read().data_start_time,
            self.stats.read().data_end_time
        );

        Ok(())
    }

    /// Check if an event should be included based on filters
    fn should_include(&self, event: &MarketEvent) -> bool {
        // Event type filter
        if !self.config.event_types.should_include(event) {
            return false;
        }

        // Time range filter
        if let Some(ts) = event.timestamp() {
            if let Some(start) = self.config.start_time {
                if ts < start {
                    return false;
                }
            }
            if let Some(end) = self.config.end_time {
                if ts > end {
                    return false;
                }
            }
        }

        // Symbol filter
        if let Some(ref symbols) = self.config.symbols {
            if let Some(event_symbol) = event.symbol() {
                if !symbols.iter().any(|s| s == event_symbol) {
                    return false;
                }
            }
        }

        // Exchange filter
        if let Some(ref exchanges) = self.config.exchanges {
            if let Some(event_exchange) = event.exchange() {
                if !exchanges.iter().any(|e| e == event_exchange) {
                    return false;
                }
            }
        }

        true
    }

    /// Subscribe to replay events
    pub fn subscribe(&self) -> broadcast::Receiver<MarketEvent> {
        self.event_tx.subscribe()
    }

    /// Get current statistics
    pub fn stats(&self) -> ReplayStats {
        self.stats.read().clone()
    }

    /// Get current state
    pub fn state(&self) -> ReplayState {
        *self.state.read()
    }

    /// Check if replay is running
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Check if replay is paused
    pub fn is_paused(&self) -> bool {
        self.paused.load(Ordering::SeqCst)
    }

    /// Start replay
    pub async fn start(&mut self) -> Result<(), ReplayError> {
        if self.events.is_empty() {
            return Err(ReplayError::LoadFailed("No events loaded".to_string()));
        }

        if self.running.load(Ordering::SeqCst) {
            return Err(ReplayError::AlreadyRunning);
        }

        info!(
            "Starting replay of {} events at speed {}",
            self.events.len(),
            self.config.speed
        );

        self.running.store(true, Ordering::SeqCst);
        self.paused.store(false, Ordering::SeqCst);
        *self.state.write() = ReplayState::Running;

        // Record start time
        {
            let mut stats = self.stats.write();
            stats.replay_started = Some(Instant::now());
            stats.events_replayed = 0;
        }

        // Run the replay loop
        self.replay_loop().await?;

        Ok(())
    }

    /// Main replay loop
    async fn replay_loop(&mut self) -> Result<(), ReplayError> {
        let total_events = self.events.len() as u64;
        let mut last_event_time: Option<DateTime<Utc>> = None;
        let start_instant = Instant::now();

        while self.running.load(Ordering::SeqCst) {
            // Check for pause
            while self.paused.load(Ordering::SeqCst) && self.running.load(Ordering::SeqCst) {
                sleep(std::time::Duration::from_millis(10)).await;
            }

            if !self.running.load(Ordering::SeqCst) {
                break;
            }

            let pos = self.position.load(Ordering::SeqCst);
            if pos >= total_events {
                // End of data
                *self.state.write() = ReplayState::Completed;
                self.running.store(false, Ordering::SeqCst);

                // Send end of data marker
                let _ = self.event_tx.send(MarketEvent::EndOfData);
                break;
            }

            let historical = &self.events[pos as usize];
            let event = &historical.event;

            // Calculate delay based on replay speed
            if let ReplaySpeed::RealTime | ReplaySpeed::Multiplier(_) = self.config.speed {
                if let (Some(last_ts), Some(current_ts)) = (last_event_time, event.timestamp()) {
                    let time_diff = current_ts - last_ts;
                    if time_diff > Duration::zero() {
                        let delay_ms = match self.config.speed {
                            ReplaySpeed::RealTime => time_diff.num_milliseconds() as u64,
                            ReplaySpeed::Multiplier(m) => {
                                (time_diff.num_milliseconds() as f64 / m) as u64
                            }
                            _ => 0,
                        };

                        if delay_ms > 0 {
                            sleep(std::time::Duration::from_millis(delay_ms)).await;
                        }
                    }
                }
            } else if let ReplaySpeed::EventsPerSecond(eps) = self.config.speed {
                // Fixed rate
                let delay_us = 1_000_000 / eps as u64;
                sleep(std::time::Duration::from_micros(delay_us)).await;
            }
            // ReplaySpeed::Maximum has no delay

            // Broadcast event
            if self.event_tx.send(event.clone()).is_err() {
                // No receivers, but that's okay
            }

            // Update last event time
            last_event_time = event.timestamp();

            // Update statistics
            {
                let mut stats = self.stats.write();
                stats.events_replayed += 1;
                stats.current_index = pos + 1;
                stats.current_time = event.timestamp();

                // Update rate calculation
                let elapsed = start_instant.elapsed().as_secs_f64();
                stats.elapsed_seconds = elapsed;
                if elapsed > 0.0 {
                    stats.events_per_second = stats.events_replayed as f64 / elapsed;
                }
            }

            // Progress logging
            if self.config.verbose && (pos + 1) % self.config.progress_interval == 0 {
                let stats = self.stats.read();
                debug!(
                    "Replay progress: {:.1}% ({}/{} events, {:.0} events/sec)",
                    stats.progress_pct(),
                    stats.events_replayed,
                    stats.total_events,
                    stats.events_per_second
                );
            }

            // Advance position
            self.position.fetch_add(1, Ordering::SeqCst);
        }

        info!(
            "Replay finished: {} events in {:.2}s ({:.0} events/sec)",
            self.stats.read().events_replayed,
            self.stats.read().elapsed_seconds,
            self.stats.read().events_per_second
        );

        Ok(())
    }

    /// Pause replay
    pub fn pause(&self) {
        if self.running.load(Ordering::SeqCst) {
            self.paused.store(true, Ordering::SeqCst);
            *self.state.write() = ReplayState::Paused;
            info!("Replay paused");
        }
    }

    /// Resume replay
    pub fn resume(&self) {
        if self.paused.load(Ordering::SeqCst) {
            self.paused.store(false, Ordering::SeqCst);
            *self.state.write() = ReplayState::Running;
            info!("Replay resumed");
        }
    }

    /// Stop replay
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
        self.paused.store(false, Ordering::SeqCst);
        *self.state.write() = ReplayState::Stopped;
        info!("Replay stopped");
    }

    /// Seek to a specific timestamp
    pub fn seek_to_time(&self, target: DateTime<Utc>) -> Result<u64, ReplayError> {
        // Binary search for the target time
        let result = self.events.binary_search_by(|h| match h.event.timestamp() {
            Some(ts) => ts.cmp(&target),
            None => std::cmp::Ordering::Less,
        });

        let new_pos = match result {
            Ok(pos) => pos,
            Err(pos) => pos.saturating_sub(1),
        };

        self.position.store(new_pos as u64, Ordering::SeqCst);

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.current_index = new_pos as u64;
            if new_pos < self.events.len() {
                stats.current_time = self.events[new_pos].event.timestamp();
            }
        }

        info!("Seeked to position {} (target: {})", new_pos, target);
        Ok(new_pos as u64)
    }

    /// Seek to a specific event index
    pub fn seek_to_index(&self, index: u64) -> Result<(), ReplayError> {
        if index as usize >= self.events.len() {
            return Err(ReplayError::SeekFailed(format!(
                "Index {} out of range (max: {})",
                index,
                self.events.len()
            )));
        }

        self.position.store(index, Ordering::SeqCst);

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.current_index = index;
            stats.current_time = self.events[index as usize].event.timestamp();
        }

        info!("Seeked to index {}", index);
        Ok(())
    }

    /// Get the current position
    pub fn current_position(&self) -> u64 {
        self.position.load(Ordering::SeqCst)
    }

    /// Get total event count
    pub fn total_events(&self) -> u64 {
        self.events.len() as u64
    }

    /// Reset to beginning
    pub fn reset(&self) {
        self.position.store(0, Ordering::SeqCst);
        self.running.store(false, Ordering::SeqCst);
        self.paused.store(false, Ordering::SeqCst);
        *self.state.write() = ReplayState::Idle;

        let mut stats = self.stats.write();
        stats.events_replayed = 0;
        stats.current_index = 0;
        stats.current_time = stats.data_start_time;
        stats.elapsed_seconds = 0.0;
        stats.events_per_second = 0.0;
        stats.replay_started = None;

        info!("Replay reset to beginning");
    }

    /// Get event at a specific index
    pub fn get_event(&self, index: u64) -> Option<&MarketEvent> {
        self.events.get(index as usize).map(|h| &h.event)
    }

    /// Get configuration
    pub fn config(&self) -> &ReplayConfig {
        &self.config
    }

    /// Update replay speed
    pub fn set_speed(&mut self, speed: ReplaySpeed) {
        self.config.speed = speed;
        info!("Replay speed changed to {}", speed);
    }

    /// Load tick data from QuestDB
    ///
    /// Queries the QuestDB REST API to load historical tick data.
    ///
    /// # Arguments
    ///
    /// * `host` - QuestDB host (e.g., "localhost")
    /// * `port` - QuestDB HTTP port (default: 9000)
    /// * `table` - Table name (e.g., "fks_ticks")
    /// * `start_time` - Optional start time filter
    /// * `end_time` - Optional end time filter
    /// * `symbols` - Optional symbol filter
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut engine = ReplayEngine::new(ReplayConfig::default());
    /// engine.load_from_questdb(
    ///     "localhost",
    ///     9000,
    ///     "fks_ticks",
    ///     Some(start),
    ///     Some(end),
    ///     Some(vec!["BTC/USDT".to_string()]),
    /// ).await?;
    /// ```
    pub async fn load_from_questdb(
        &mut self,
        host: &str,
        port: u16,
        table: &str,
        start_time: Option<DateTime<Utc>>,
        end_time: Option<DateTime<Utc>>,
        symbols: Option<Vec<String>>,
    ) -> Result<usize, ReplayError> {
        info!(
            "Loading data from QuestDB {}:{}, table: {}",
            host, port, table
        );

        // Build query
        let mut query = format!(
            "SELECT symbol, exchange, bid_price, ask_price, bid_size, ask_size, timestamp FROM {} WHERE 1=1",
            table
        );

        if let Some(start) = start_time {
            query.push_str(&format!(
                " AND timestamp >= '{}'",
                start.format("%Y-%m-%dT%H:%M:%S%.6fZ")
            ));
        }

        if let Some(end) = end_time {
            query.push_str(&format!(
                " AND timestamp <= '{}'",
                end.format("%Y-%m-%dT%H:%M:%S%.6fZ")
            ));
        }

        if let Some(ref syms) = symbols {
            let sym_list: Vec<String> = syms.iter().map(|s| format!("'{}'", s)).collect();
            query.push_str(&format!(" AND symbol IN ({})", sym_list.join(",")));
        }

        query.push_str(" ORDER BY timestamp ASC");

        // Execute query via HTTP
        let events = query_questdb_ticks(host, port, &query).await?;
        let count = events.len();

        // Load into replay engine
        self.load_events(events)?;

        info!("Loaded {} tick events from QuestDB", count);
        Ok(count)
    }

    /// Load trade data from QuestDB
    pub async fn load_trades_from_questdb(
        &mut self,
        host: &str,
        port: u16,
        table: &str,
        start_time: Option<DateTime<Utc>>,
        end_time: Option<DateTime<Utc>>,
        symbols: Option<Vec<String>>,
    ) -> Result<usize, ReplayError> {
        info!(
            "Loading trades from QuestDB {}:{}, table: {}",
            host, port, table
        );

        // Build query
        let mut query = format!(
            "SELECT symbol, exchange, price, size, side, trade_id, timestamp FROM {} WHERE 1=1",
            table
        );

        if let Some(start) = start_time {
            query.push_str(&format!(
                " AND timestamp >= '{}'",
                start.format("%Y-%m-%dT%H:%M:%S%.6fZ")
            ));
        }

        if let Some(end) = end_time {
            query.push_str(&format!(
                " AND timestamp <= '{}'",
                end.format("%Y-%m-%dT%H:%M:%S%.6fZ")
            ));
        }

        if let Some(ref syms) = symbols {
            let sym_list: Vec<String> = syms.iter().map(|s| format!("'{}'", s)).collect();
            query.push_str(&format!(" AND symbol IN ({})", sym_list.join(",")));
        }

        query.push_str(" ORDER BY timestamp ASC");

        // Execute query via HTTP
        let events = query_questdb_trades(host, port, &query).await?;
        let count = events.len();

        // Load into replay engine
        self.load_events(events)?;

        info!("Loaded {} trade events from QuestDB", count);
        Ok(count)
    }

    /// Load both ticks and trades from QuestDB and merge them
    pub async fn load_all_from_questdb(
        &mut self,
        host: &str,
        port: u16,
        ticks_table: &str,
        trades_table: &str,
        start_time: Option<DateTime<Utc>>,
        end_time: Option<DateTime<Utc>>,
        symbols: Option<Vec<String>>,
    ) -> Result<usize, ReplayError> {
        info!("Loading ticks and trades from QuestDB");

        let mut all_events = Vec::new();

        // Load ticks
        if self.config.event_types.ticks {
            let tick_events = load_ticks_from_questdb(
                host,
                port,
                ticks_table,
                start_time,
                end_time,
                symbols.clone(),
            )
            .await?;
            info!("Loaded {} ticks", tick_events.len());
            all_events.extend(tick_events);
        }

        // Load trades
        if self.config.event_types.trades {
            let trade_events =
                load_trades_from_questdb(host, port, trades_table, start_time, end_time, symbols)
                    .await?;
            info!("Loaded {} trades", trade_events.len());
            all_events.extend(trade_events);
        }

        let count = all_events.len();
        self.load_events(all_events)?;

        info!("Loaded {} total events from QuestDB", count);
        Ok(count)
    }
}

/// Configuration for QuestDB data loading
#[derive(Debug, Clone)]
pub struct QuestDBLoaderConfig {
    /// QuestDB host
    pub host: String,
    /// QuestDB HTTP port (default: 9000)
    pub port: u16,
    /// Ticks table name
    pub ticks_table: String,
    /// Trades table name
    pub trades_table: String,
    /// Start time filter
    pub start_time: Option<DateTime<Utc>>,
    /// End time filter
    pub end_time: Option<DateTime<Utc>>,
    /// Symbol filter
    pub symbols: Option<Vec<String>>,
    /// Exchange filter
    pub exchanges: Option<Vec<String>>,
}

impl Default for QuestDBLoaderConfig {
    fn default() -> Self {
        Self {
            host: "localhost".to_string(),
            port: 9000,
            ticks_table: "fks_ticks".to_string(),
            trades_table: "fks_trades".to_string(),
            start_time: None,
            end_time: None,
            symbols: None,
            exchanges: None,
        }
    }
}

impl QuestDBLoaderConfig {
    /// Create a new loader config
    pub fn new(host: &str, port: u16) -> Self {
        Self {
            host: host.to_string(),
            port,
            ..Default::default()
        }
    }

    /// Set ticks table name
    pub fn with_ticks_table(mut self, table: &str) -> Self {
        self.ticks_table = table.to_string();
        self
    }

    /// Set trades table name
    pub fn with_trades_table(mut self, table: &str) -> Self {
        self.trades_table = table.to_string();
        self
    }

    /// Set time range
    pub fn with_time_range(mut self, start: DateTime<Utc>, end: DateTime<Utc>) -> Self {
        self.start_time = Some(start);
        self.end_time = Some(end);
        self
    }

    /// Set symbol filter
    pub fn with_symbols(mut self, symbols: Vec<String>) -> Self {
        self.symbols = Some(symbols);
        self
    }

    /// Set exchange filter
    pub fn with_exchanges(mut self, exchanges: Vec<String>) -> Self {
        self.exchanges = Some(exchanges);
        self
    }
}

/// Query QuestDB for tick data
async fn query_questdb_ticks(
    host: &str,
    port: u16,
    query: &str,
) -> Result<Vec<MarketEvent>, ReplayError> {
    let encoded_query: String = form_urlencoded::Serializer::new(String::new())
        .append_pair("query", query)
        .append_pair("fmt", "json")
        .finish();
    let url = format!("http://{}:{}/exec?{}", host, port, encoded_query);

    debug!("QuestDB query URL: {}", url);

    let client = reqwest::Client::new();
    let response = client
        .get(&url)
        .send()
        .await
        .map_err(|e| ReplayError::QueryError(format!("HTTP request failed: {}", e)))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(ReplayError::QueryError(format!(
            "Query failed with status {}: {}",
            status, body
        )));
    }

    let body: serde_json::Value = response
        .json()
        .await
        .map_err(|e| ReplayError::ParseError(format!("JSON parse error: {}", e)))?;

    parse_tick_response(&body)
}

/// Query QuestDB for trade data
async fn query_questdb_trades(
    host: &str,
    port: u16,
    query: &str,
) -> Result<Vec<MarketEvent>, ReplayError> {
    let encoded_query: String = form_urlencoded::Serializer::new(String::new())
        .append_pair("query", query)
        .append_pair("fmt", "json")
        .finish();
    let url = format!("http://{}:{}/exec?{}", host, port, encoded_query);

    debug!("QuestDB query URL: {}", url);

    let client = reqwest::Client::new();
    let response = client
        .get(&url)
        .send()
        .await
        .map_err(|e| ReplayError::QueryError(format!("HTTP request failed: {}", e)))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(ReplayError::QueryError(format!(
            "Query failed with status {}: {}",
            status, body
        )));
    }

    let body: serde_json::Value = response
        .json()
        .await
        .map_err(|e| ReplayError::ParseError(format!("JSON parse error: {}", e)))?;

    parse_trade_response(&body)
}

/// Parse QuestDB tick response JSON
fn parse_tick_response(body: &serde_json::Value) -> Result<Vec<MarketEvent>, ReplayError> {
    let dataset = body["dataset"]
        .as_array()
        .ok_or_else(|| ReplayError::ParseError("Missing 'dataset' field".to_string()))?;

    let mut events = Vec::with_capacity(dataset.len());

    for row in dataset {
        let row = row
            .as_array()
            .ok_or_else(|| ReplayError::ParseError("Invalid row format".to_string()))?;

        if row.len() < 7 {
            warn!("Skipping row with insufficient columns: {:?}", row);
            continue;
        }

        let symbol = row[0].as_str().unwrap_or("").to_string();
        let exchange = row[1].as_str().unwrap_or("").to_string();

        let bid_price = parse_decimal(&row[2])?;
        let ask_price = parse_decimal(&row[3])?;
        let bid_size = parse_decimal(&row[4])?;
        let ask_size = parse_decimal(&row[5])?;
        let timestamp = parse_timestamp(&row[6])?;

        let tick = TickData::new(
            &symbol, &exchange, bid_price, ask_price, bid_size, ask_size, timestamp,
        );

        events.push(MarketEvent::Tick(tick));
    }

    Ok(events)
}

/// Parse QuestDB trade response JSON
fn parse_trade_response(body: &serde_json::Value) -> Result<Vec<MarketEvent>, ReplayError> {
    let dataset = body["dataset"]
        .as_array()
        .ok_or_else(|| ReplayError::ParseError("Missing 'dataset' field".to_string()))?;

    let mut events = Vec::with_capacity(dataset.len());

    for row in dataset {
        let row = row
            .as_array()
            .ok_or_else(|| ReplayError::ParseError("Invalid row format".to_string()))?;

        if row.len() < 7 {
            warn!("Skipping row with insufficient columns: {:?}", row);
            continue;
        }

        let symbol = row[0].as_str().unwrap_or("").to_string();
        let exchange = row[1].as_str().unwrap_or("").to_string();
        let price = parse_decimal(&row[2])?;
        let size = parse_decimal(&row[3])?;
        let side_str = row[4].as_str().unwrap_or("unknown");
        let trade_id = row[5].as_str().unwrap_or("").to_string();
        let timestamp = parse_timestamp(&row[6])?;

        let side = match side_str.to_lowercase().as_str() {
            "buy" | "b" => TradeSide::Buy,
            "sell" | "s" => TradeSide::Sell,
            _ => TradeSide::Unknown,
        };

        let trade = TradeData {
            symbol,
            exchange,
            price,
            size,
            side,
            trade_id,
            timestamp,
        };

        events.push(MarketEvent::Trade(trade));
    }

    Ok(events)
}

/// Parse a decimal value from JSON
fn parse_decimal(value: &serde_json::Value) -> Result<Decimal, ReplayError> {
    if let Some(n) = value.as_f64() {
        Decimal::from_str(&n.to_string())
            .map_err(|e| ReplayError::ParseError(format!("Invalid decimal: {}", e)))
    } else if let Some(s) = value.as_str() {
        Decimal::from_str(s)
            .map_err(|e| ReplayError::ParseError(format!("Invalid decimal string: {}", e)))
    } else {
        Err(ReplayError::ParseError(format!(
            "Cannot parse decimal from: {:?}",
            value
        )))
    }
}

/// Parse a timestamp from JSON (QuestDB returns microseconds since epoch)
fn parse_timestamp(value: &serde_json::Value) -> Result<DateTime<Utc>, ReplayError> {
    if let Some(micros) = value.as_i64() {
        // QuestDB returns microseconds since epoch
        let secs = micros / 1_000_000;
        let nsecs = ((micros % 1_000_000) * 1000) as u32;
        Ok(Utc
            .timestamp_opt(secs, nsecs)
            .single()
            .unwrap_or_else(Utc::now))
    } else if let Some(s) = value.as_str() {
        // Try to parse as ISO string
        DateTime::parse_from_rfc3339(s)
            .map(|dt| dt.with_timezone(&Utc))
            .or_else(|_| {
                // Try other formats
                chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S%.fZ")
                    .map(|ndt| Utc.from_utc_datetime(&ndt))
            })
            .map_err(|e| ReplayError::ParseError(format!("Invalid timestamp: {}", e)))
    } else {
        Err(ReplayError::ParseError(format!(
            "Cannot parse timestamp from: {:?}",
            value
        )))
    }
}

/// Helper function to load ticks from QuestDB
async fn load_ticks_from_questdb(
    host: &str,
    port: u16,
    table: &str,
    start_time: Option<DateTime<Utc>>,
    end_time: Option<DateTime<Utc>>,
    symbols: Option<Vec<String>>,
) -> Result<Vec<MarketEvent>, ReplayError> {
    let mut query = format!(
        "SELECT symbol, exchange, bid_price, ask_price, bid_size, ask_size, timestamp FROM {} WHERE 1=1",
        table
    );

    if let Some(start) = start_time {
        query.push_str(&format!(
            " AND timestamp >= '{}'",
            start.format("%Y-%m-%dT%H:%M:%S%.6fZ")
        ));
    }

    if let Some(end) = end_time {
        query.push_str(&format!(
            " AND timestamp <= '{}'",
            end.format("%Y-%m-%dT%H:%M:%S%.6fZ")
        ));
    }

    if let Some(ref syms) = symbols {
        let sym_list: Vec<String> = syms.iter().map(|s| format!("'{}'", s)).collect();
        query.push_str(&format!(" AND symbol IN ({})", sym_list.join(",")));
    }

    query.push_str(" ORDER BY timestamp ASC");

    query_questdb_ticks(host, port, &query).await
}

/// Helper function to load trades from QuestDB
async fn load_trades_from_questdb(
    host: &str,
    port: u16,
    table: &str,
    start_time: Option<DateTime<Utc>>,
    end_time: Option<DateTime<Utc>>,
    symbols: Option<Vec<String>>,
) -> Result<Vec<MarketEvent>, ReplayError> {
    let mut query = format!(
        "SELECT symbol, exchange, price, size, side, trade_id, timestamp FROM {} WHERE 1=1",
        table
    );

    if let Some(start) = start_time {
        query.push_str(&format!(
            " AND timestamp >= '{}'",
            start.format("%Y-%m-%dT%H:%M:%S%.6fZ")
        ));
    }

    if let Some(end) = end_time {
        query.push_str(&format!(
            " AND timestamp <= '{}'",
            end.format("%Y-%m-%dT%H:%M:%S%.6fZ")
        ));
    }

    if let Some(ref syms) = symbols {
        let sym_list: Vec<String> = syms.iter().map(|s| format!("'{}'", s)).collect();
        query.push_str(&format!(" AND symbol IN ({})", sym_list.join(",")));
    }

    query.push_str(" ORDER BY timestamp ASC");

    query_questdb_trades(host, port, &query).await
}

#[cfg(test)]
mod tests {
    use super::super::data_feed::TickData;
    use super::*;
    use rust_decimal::Decimal;
    use rust_decimal_macros::dec;

    fn create_test_ticks(count: usize) -> Vec<MarketEvent> {
        let base_time = Utc::now();
        (0..count)
            .map(|i| {
                let tick = TickData::new(
                    "BTC/USDT",
                    "kraken",
                    dec!(50000.0) + Decimal::from(i as i64),
                    dec!(50001.0) + Decimal::from(i as i64),
                    dec!(1.0),
                    dec!(1.0),
                    base_time + Duration::milliseconds(i as i64 * 100),
                );
                MarketEvent::Tick(tick)
            })
            .collect()
    }

    #[test]
    fn test_replay_config() {
        let config = ReplayConfig::fast()
            .with_speed(ReplaySpeed::Multiplier(10.0))
            .with_symbols(vec!["BTC/USDT".to_string()])
            .with_verbose(true);

        assert_eq!(config.speed, ReplaySpeed::Multiplier(10.0));
        assert_eq!(config.symbols, Some(vec!["BTC/USDT".to_string()]));
        assert!(config.verbose);
    }

    #[test]
    fn test_replay_speed_display() {
        assert_eq!(format!("{}", ReplaySpeed::RealTime), "1x (real-time)");
        assert_eq!(format!("{}", ReplaySpeed::Multiplier(10.0)), "10x");
        assert_eq!(format!("{}", ReplaySpeed::Maximum), "max");
        assert_eq!(
            format!("{}", ReplaySpeed::EventsPerSecond(1000)),
            "1000 events/sec"
        );
    }

    #[test]
    fn test_event_type_filter() {
        let tick = MarketEvent::Tick(TickData::new(
            "BTC/USDT",
            "kraken",
            dec!(50000.0),
            dec!(50001.0),
            dec!(1.0),
            dec!(1.0),
            Utc::now(),
        ));

        let all_filter = EventTypeFilter::default();
        assert!(all_filter.should_include(&tick));

        let trades_only = EventTypeFilter::trades_only();
        assert!(!trades_only.should_include(&tick));

        let ticks_only = EventTypeFilter::ticks_only();
        assert!(ticks_only.should_include(&tick));
    }

    #[test]
    fn test_replay_stats() {
        let mut stats = ReplayStats::default();
        stats.total_events = 1000;
        stats.current_index = 500;

        assert_eq!(stats.progress_pct(), 50.0);

        stats.events_per_second = 100.0;
        let remaining = stats.estimated_remaining_seconds().unwrap();
        assert_eq!(remaining, 5.0); // 500 remaining / 100 eps = 5 seconds
    }

    #[test]
    fn test_replay_engine_creation() {
        let config = ReplayConfig::default();
        let engine = ReplayEngine::new(config);

        assert_eq!(engine.state(), ReplayState::Idle);
        assert!(!engine.is_running());
        assert!(!engine.is_paused());
        assert_eq!(engine.total_events(), 0);
    }

    #[test]
    fn test_load_events() {
        let config = ReplayConfig::default();
        let mut engine = ReplayEngine::new(config);

        let events = create_test_ticks(100);
        engine.load_events(events).unwrap();

        assert_eq!(engine.total_events(), 100);

        let stats = engine.stats();
        assert_eq!(stats.total_events, 100);
        assert!(stats.data_start_time.is_some());
        assert!(stats.data_end_time.is_some());
        assert!(stats.symbols.contains(&"BTC/USDT".to_string()));
        assert!(stats.exchanges.contains(&"kraken".to_string()));
    }

    #[test]
    fn test_symbol_filter() {
        let config = ReplayConfig::default().with_symbols(vec!["ETH/USDT".to_string()]);
        let mut engine = ReplayEngine::new(config);

        let events = create_test_ticks(100); // BTC/USDT ticks
        engine.load_events(events).unwrap();

        // All events should be filtered out
        assert_eq!(engine.total_events(), 0);
    }

    #[test]
    fn test_seek() {
        let config = ReplayConfig::default();
        let mut engine = ReplayEngine::new(config);

        let events = create_test_ticks(100);
        engine.load_events(events).unwrap();

        // Seek to index
        engine.seek_to_index(50).unwrap();
        assert_eq!(engine.current_position(), 50);

        // Invalid index should fail
        assert!(engine.seek_to_index(200).is_err());

        // Reset
        engine.reset();
        assert_eq!(engine.current_position(), 0);
        assert_eq!(engine.state(), ReplayState::Idle);
    }

    #[test]
    fn test_get_event() {
        let config = ReplayConfig::default();
        let mut engine = ReplayEngine::new(config);

        let events = create_test_ticks(10);
        engine.load_events(events).unwrap();

        let event = engine.get_event(0);
        assert!(event.is_some());
        assert!(event.unwrap().is_tick());

        // Out of bounds
        assert!(engine.get_event(100).is_none());
    }

    #[tokio::test]
    async fn test_replay_execution() {
        let config = ReplayConfig::fast();
        let mut engine = ReplayEngine::new(config);

        let events = create_test_ticks(10);
        engine.load_events(events).unwrap();

        // Subscribe before starting
        let mut rx = engine.subscribe();

        // Start replay
        engine.start().await.unwrap();

        // Should have received events + EndOfData
        let mut received_count = 0;
        while let Ok(event) = rx.try_recv() {
            if matches!(event, MarketEvent::EndOfData) {
                break;
            }
            received_count += 1;
        }

        assert_eq!(received_count, 10);
        assert_eq!(engine.state(), ReplayState::Completed);
    }

    #[test]
    fn test_pause_resume_stop() {
        let config = ReplayConfig::default();
        let engine = ReplayEngine::new(config);

        // Initial state
        assert!(!engine.is_paused());

        // Pause (no effect when not running)
        engine.pause();
        // Won't actually pause since not running

        // Stop
        engine.stop();
        assert_eq!(engine.state(), ReplayState::Stopped);
    }
}
